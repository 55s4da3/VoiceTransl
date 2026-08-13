"""Local-network bridge used by PixelPlayer.

The HTTP/UDP threads in this module never touch Qt.  Callbacks supplied by the
desktop controller operate on immutable dictionaries and thread-safe queues.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import mimetypes
import os
import secrets
import shutil
import socket
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse


API_VERSION = 1
DISCOVERY_PORT = 49731
DEFAULT_HTTP_PORT = 49732
DISCOVERY_REQUEST = b"VOICETRANSL_DISCOVER_V1"
JOB_TTL_SECONDS = 24 * 60 * 60
PAIR_CODE_TTL_SECONDS = 10 * 60
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _safe_filename(value: str) -> str:
    name = Path(str(value or "audio.bin")).name
    name = "".join("_" if ord(ch) < 32 or ch in '<>:"/\\|?*' else ch for ch in name)
    name = name.strip(" .")[:180]
    return name or "audio.bin"


def _is_private_client(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
        return address.is_private or address.is_loopback or address.is_link_local
    except ValueError:
        return False


class LanJobRegistry:
    """Persistent public job state plus in-memory immutable task snapshots."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._last_persist: dict[str, float] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        now = time.time()
        for state_file in self.root.glob("*/state.json"):
            try:
                job = json.loads(state_file.read_text(encoding="utf-8"))
                job_id = str(job["id"])
                if now - float(job.get("updated_at", 0)) > JOB_TTL_SECONDS:
                    shutil.rmtree(state_file.parent, ignore_errors=True)
                    continue
                if job.get("state") in {"created", "uploading", "queued", "running"}:
                    job["state"] = "failed"
                    job["error"] = "VoiceTransl restarted before the task completed"
                    job["updated_at"] = now
                    _atomic_json(state_file, job)
                self._jobs[job_id] = job
            except Exception:
                continue

    def cleanup(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        with self._lock:
            expired = [
                job_id for job_id, job in self._jobs.items()
                if float(job.get("updated_at", 0)) < cutoff
            ]
            for job_id in expired:
                self._jobs.pop(job_id, None)
                self._snapshots.pop(job_id, None)
                shutil.rmtree(self.root / job_id, ignore_errors=True)

    def create(
        self,
        *,
        device_id: str,
        filename: str,
        size: int,
        profile: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        self.cleanup()
        job_id = uuid.uuid4().hex
        now = time.time()
        directory = self.root / job_id
        (directory / "output").mkdir(parents=True, exist_ok=False)
        job = {
            "id": job_id,
            "device_id": device_id,
            "filename": _safe_filename(filename),
            "size": int(size),
            "received": 0,
            "state": "created",
            "phase": "waiting_upload",
            "current": 0,
            "total": max(1, int(size)),
            "message": "",
            "error": "",
            "profile": profile,
            "artifacts": [],
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._snapshots[job_id] = dict(snapshot)
            self._persist_locked(job_id)
        return self.public(job_id) or {}

    def _persist_locked(self, job_id: str) -> None:
        _atomic_json(self.root / job_id / "state.json", self._jobs[job_id])

    def public(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return json.loads(json.dumps(job, ensure_ascii=False))

    def belongs_to(self, job_id: str, device_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.get("device_id") == device_id)

    def paths(self, job_id: str) -> tuple[Path, Path]:
        with self._lock:
            job = self._jobs[job_id]
            filename = job["filename"]
        directory = (self.root / job_id).resolve()
        if directory.parent != self.root:
            raise ValueError("invalid job path")
        return directory / filename, directory / "output"

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._snapshots.get(job_id)
            return dict(value) if value is not None else None

    def update(self, job_id: str, **changes: Any) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.update(changes)
            job["updated_at"] = time.time()
            now = time.monotonic()
            force = bool(
                {"state", "artifacts", "error"}.intersection(changes)
                or now - self._last_persist.get(job_id, 0.0) >= 0.25
            )
            if force:
                self._persist_locked(job_id)
                self._last_persist[job_id] = now
        return self.public(job_id)

    def update_progress(self, job_id: str, target: str, text: str) -> None:
        changes: dict[str, Any] = {}
        if target == "status":
            changes["message"] = str(text)[-1000:]
        elif target == "progress":
            try:
                event = json.loads(text)
            except Exception:
                return
            if event.get("kind") == "stage":
                changes.update(
                    phase=str(event.get("phase", "")),
                    current=max(0, int(event.get("current", 0))),
                    total=max(1, int(event.get("total", 1))),
                )
            elif event.get("kind") == "files":
                changes["files"] = {
                    "completed": max(0, int(event.get("completed", 0))),
                    "total": max(0, int(event.get("total", 0))),
                }
        if changes:
            self.update(job_id, **changes)

    def set_artifacts(self, job_id: str, artifacts: list[dict[str, Any]]) -> None:
        self.update(
            job_id,
            artifacts=artifacts,
            state="succeeded",
            phase="completed",
            current=1,
            total=1,
            error="",
        )

    def artifact_path(self, job_id: str, kind: str) -> Path | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            match = next((x for x in job.get("artifacts", []) if x.get("kind") == kind), None)
            if not match:
                return None
            candidate = (self.root / job_id / "output" / match["filename"]).resolve()
            output = (self.root / job_id / "output").resolve()
            return candidate if candidate.parent == output and candidate.is_file() else None


class _ForwardingQueue:
    def __init__(self, delegate: Any, registry: LanJobRegistry, job_id: str):
        self._delegate = delegate
        self._registry = registry
        self._job_id = job_id

    def put(self, target: str, text: str) -> None:
        self._registry.update_progress(self._job_id, target, text)
        self._delegate.put(target, text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class LanService:
    def __init__(
        self,
        *,
        root: Path,
        profile_provider: Callable[[], dict[str, Any]],
        job_ready: Callable[[str], None],
        cancel_job: Callable[[str], None],
        device_name: str = "VoiceTransl",
        port: int = DEFAULT_HTTP_PORT,
        discovery_port: int = DISCOVERY_PORT,
        state_dir: Path | None = None,
    ):
        self.registry = LanJobRegistry(root)
        self.profile_provider = profile_provider
        self.job_ready = job_ready
        self.cancel_job = cancel_job
        self.device_name = device_name.strip() or "VoiceTransl"
        self.port = int(port)
        self.discovery_port = int(discovery_port)
        self._state_dir = (state_dir or Path.cwd()).resolve()
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self.server_id = self._load_server_id()
        self._devices_path = self._state_dir / "lan_devices.json"
        self._devices_lock = threading.RLock()
        self._devices = self._load_devices()
        self._pair_attempts: dict[str, list[float]] = {}
        self._pair_code = ""
        self._pair_code_expires = 0.0
        self._http: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._udp_socket: socket.socket | None = None
        self._udp_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _load_server_id(self) -> str:
        path = self._state_dir / "lan_server_id.txt"
        try:
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            pass
        value = uuid.uuid4().hex
        path.write_text(value, encoding="utf-8")
        return value

    def _load_devices(self) -> dict[str, dict[str, str]]:
        try:
            value = json.loads(self._devices_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _save_devices(self) -> None:
        with self._devices_lock:
            _atomic_json(self._devices_path, self._devices)

    @property
    def running(self) -> bool:
        return self._http is not None

    @property
    def pair_code(self) -> str:
        if time.time() >= self._pair_code_expires:
            self.rotate_pair_code()
        return self._pair_code

    def rotate_pair_code(self) -> str:
        self._pair_code = f"{secrets.randbelow(1_000_000):06d}"
        self._pair_code_expires = time.time() + PAIR_CODE_TTL_SECONDS
        return self._pair_code

    def paired_devices(self) -> list[dict[str, str]]:
        with self._devices_lock:
            return [
                {"id": key, "name": value.get("name", key)}
                for key, value in self._devices.items()
            ]

    def revoke(self, device_id: str) -> None:
        with self._devices_lock:
            self._devices.pop(device_id, None)
            self._save_devices()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.registry.cleanup()
        self.rotate_pair_code()
        handler = self._build_handler()
        self._http = ThreadingHTTPServer(("0.0.0.0", self.port), handler)
        self._http.daemon_threads = True
        self.port = int(self._http.server_port)
        self._http_thread = threading.Thread(
            target=self._http.serve_forever, name="voicetransl-lan-http", daemon=True
        )
        self._http_thread.start()
        self._udp_thread = threading.Thread(
            target=self._serve_discovery, name="voicetransl-lan-discovery", daemon=True
        )
        self._udp_thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock, self._udp_socket = self._udp_socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        server, self._http = self._http, None
        if server is not None:
            server.shutdown()
            server.server_close()
        for thread in (self._udp_thread, self._http_thread):
            if thread and thread is not threading.current_thread():
                thread.join(timeout=2)
        self._udp_thread = None
        self._http_thread = None

    def forwarding_queue(self, delegate: Any, job_id: str) -> Any:
        return _ForwardingQueue(delegate, self.registry, job_id)

    def _serve_discovery(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("", self.discovery_port))
        sock.settimeout(0.5)
        self._udp_socket = sock
        while not self._stop.is_set():
            try:
                payload, address = sock.recvfrom(1024)
                if payload.strip() != DISCOVERY_REQUEST or not _is_private_client(address[0]):
                    continue
                response = json.dumps(
                    {
                        "api_version": API_VERSION,
                        "id": self.server_id,
                        "name": self.device_name,
                        "port": self.port,
                        "requires_pairing": True,
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                sock.sendto(response, address)
            except socket.timeout:
                continue
            except OSError:
                break

    def _issue_token(self, device_id: str, device_name: str) -> str:
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._devices_lock:
            self._devices[device_id] = {"name": device_name[:80], "token_hash": digest}
            self._save_devices()
        return token

    def _authenticate(self, authorization: str) -> tuple[str, bool]:
        if not authorization.startswith("Bearer "):
            return "", False
        token = authorization[7:].strip()
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._devices_lock:
            for device_id, entry in self._devices.items():
                if secrets.compare_digest(str(entry.get("token_hash", "")), digest):
                    return device_id, True
        return "", False

    def _pair_allowed(self, host: str) -> bool:
        now = time.time()
        attempts = [x for x in self._pair_attempts.get(host, []) if now - x < 60]
        self._pair_attempts[host] = attempts
        return len(attempts) < 5

    def _record_pair_attempt(self, host: str) -> None:
        self._pair_attempts.setdefault(host, []).append(time.time())

    def _build_handler(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "VoiceTranslLAN/1"

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _json(self, status: int, payload: Any) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _body_json(self, maximum: int = 64 * 1024) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length <= 0 or length > maximum:
                    raise ValueError("invalid request size")
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("JSON object required")
                return value

            def _private(self) -> bool:
                if _is_private_client(self.client_address[0]):
                    return True
                self._json(HTTPStatus.FORBIDDEN, {"error": "LAN clients only"})
                return False

            def _auth(self) -> str | None:
                device_id, valid = service._authenticate(self.headers.get("Authorization", ""))
                if valid:
                    return device_id
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "invalid device token"})
                return None

            def do_GET(self) -> None:  # noqa: N802
                if not self._private():
                    return
                parsed = urlparse(self.path)
                parts = [unquote(x) for x in parsed.path.strip("/").split("/") if x]
                if parts == ["api", "v1", "info"]:
                    self._json(HTTPStatus.OK, {
                        "api_version": API_VERSION, "id": service.server_id,
                        "name": service.device_name, "requires_pairing": True,
                    })
                    return
                device_id = self._auth()
                if device_id is None:
                    return
                if parts == ["api", "v1", "profile"]:
                    current = service.profile_provider()
                    self._json(HTTPStatus.OK, current.get("public", {}))
                    return
                if len(parts) == 4 and parts[:3] == ["api", "v1", "jobs"]:
                    job_id = parts[3]
                    if not service.registry.belongs_to(job_id, device_id):
                        self._json(HTTPStatus.NOT_FOUND, {"error": "job not found"})
                        return
                    self._json(HTTPStatus.OK, service.registry.public(job_id))
                    return
                if len(parts) == 6 and parts[:3] == ["api", "v1", "jobs"] and parts[4] == "artifacts":
                    job_id, kind = parts[3], parts[5]
                    if not service.registry.belongs_to(job_id, device_id):
                        self._json(HTTPStatus.NOT_FOUND, {"error": "job not found"})
                        return
                    path = service.registry.artifact_path(job_id, kind)
                    if path is None:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "artifact not found"})
                        return
                    size = path.stat().st_size
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/x-subrip")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
                    self.end_headers()
                    with path.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile, 1024 * 1024)
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if not self._private():
                    return
                parsed = urlparse(self.path)
                parts = [x for x in parsed.path.strip("/").split("/") if x]
                if parts == ["api", "v1", "pair"]:
                    host = self.client_address[0]
                    if not service._pair_allowed(host):
                        self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "too many attempts"})
                        return
                    try:
                        body = self._body_json()
                    except Exception as error:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                        return
                    service._record_pair_attempt(host)
                    code = str(body.get("code", ""))
                    if time.time() >= service._pair_code_expires or not secrets.compare_digest(code, service.pair_code):
                        self._json(HTTPStatus.UNAUTHORIZED, {"error": "invalid or expired pairing code"})
                        return
                    device_id = str(body.get("device_id", "")).strip()[:100] or uuid.uuid4().hex
                    device_name = str(body.get("device_name", "PixelPlayer")).strip() or "PixelPlayer"
                    token = service._issue_token(device_id, device_name)
                    service.rotate_pair_code()
                    self._json(HTTPStatus.OK, {"token": token, "device_id": device_id, "server_id": service.server_id})
                    return
                device_id = self._auth()
                if device_id is None:
                    return
                if parts == ["api", "v1", "jobs"]:
                    try:
                        body = self._body_json()
                        size = int(body.get("size", 0))
                        if size <= 0 or size > MAX_UPLOAD_BYTES:
                            raise ValueError("unsupported file size")
                        free = shutil.disk_usage(service.registry.root).free
                        if size + 2 * 1024 * 1024 * 1024 > free:
                            raise OSError("not enough free disk space")
                        provider = service.profile_provider()
                        public = provider.get("public", {})
                        if body.get("profile_revision") != public.get("revision"):
                            self._json(HTTPStatus.CONFLICT, {"error": "profile_changed", "profile": public})
                            return
                        if not public.get("ready", False):
                            self._json(HTTPStatus.CONFLICT, {"error": "profile_not_ready", "profile": public})
                            return
                        job = service.registry.create(
                            device_id=device_id,
                            filename=str(body.get("filename", "audio.bin")),
                            size=size,
                            profile=public,
                            snapshot=provider.get("snapshot", {}),
                        )
                        self._json(HTTPStatus.CREATED, job)
                    except (ValueError, OSError) as error:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_PUT(self) -> None:  # noqa: N802
                if not self._private():
                    return
                device_id = self._auth()
                if device_id is None:
                    return
                parts = [x for x in urlparse(self.path).path.strip("/").split("/") if x]
                if len(parts) != 5 or parts[:3] != ["api", "v1", "jobs"] or parts[4] != "audio":
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                job_id = parts[3]
                job = service.registry.public(job_id)
                if not job or not service.registry.belongs_to(job_id, device_id):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "job not found"})
                    return
                expected = int(job["size"])
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length != expected:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "Content-Length does not match job size"})
                    return
                input_path, _output = service.registry.paths(job_id)
                temporary = input_path.with_suffix(input_path.suffix + ".part")
                digest = hashlib.sha256()
                received = 0
                service.registry.update(job_id, state="uploading", phase="upload", current=0, total=expected, received=0)
                try:
                    with temporary.open("wb") as target:
                        while received < expected:
                            chunk = self.rfile.read(min(1024 * 1024, expected - received))
                            if not chunk:
                                raise ConnectionError("upload ended early")
                            target.write(chunk)
                            digest.update(chunk)
                            received += len(chunk)
                            if received == expected or received % (8 * 1024 * 1024) < len(chunk):
                                service.registry.update(job_id, current=received, received=received)
                    os.replace(temporary, input_path)
                    service.registry.update(
                        job_id, state="queued", phase="queued", current=0, total=1,
                        received=received, sha256=digest.hexdigest(),
                    )
                    service.job_ready(job_id)
                    self._json(HTTPStatus.OK, service.registry.public(job_id))
                except Exception as error:
                    temporary.unlink(missing_ok=True)
                    service.registry.update(job_id, state="failed", error=str(error), phase="upload_failed")
                    if not isinstance(error, ConnectionError):
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return

            def do_DELETE(self) -> None:  # noqa: N802
                if not self._private():
                    return
                device_id = self._auth()
                if device_id is None:
                    return
                parts = [x for x in urlparse(self.path).path.strip("/").split("/") if x]
                if len(parts) != 4 or parts[:3] != ["api", "v1", "jobs"]:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                job_id = parts[3]
                if not service.registry.belongs_to(job_id, device_id):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "job not found"})
                    return
                service.cancel_job(job_id)
                self._json(HTTPStatus.ACCEPTED, {"id": job_id, "state": "cancelling"})

        return Handler


def build_artifact(path: Path, kind: str, language: str = "") -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "kind": kind,
        "language": language,
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
