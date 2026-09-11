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
import re
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
from urllib.parse import parse_qs, quote, unquote, urlparse


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


def _content_disposition(value: str, *, attachment: bool) -> str:
    """Build an ASCII-only header while preserving Unicode through RFC 5987."""
    name = _safe_filename(value)
    suffix = Path(name).suffix
    fallback_suffix = suffix if suffix.isascii() and re.fullmatch(r"\.[A-Za-z0-9]{1,12}", suffix) else ".bin"
    disposition = "attachment" if attachment else "inline"
    return (
        f'{disposition}; filename="download{fallback_suffix}"; '
        f"filename*=UTF-8''{quote(name, safe='')}"
    )


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
        media_library: Any | None = None,
        metadata_client: Any | None = None,
        allowed_networks: list[str] | tuple[str, ...] | None = None,
    ):
        self.registry = LanJobRegistry(root)
        self.profile_provider = profile_provider
        self.job_ready = job_ready
        self.cancel_job = cancel_job
        self.device_name = device_name.strip() or "VoiceTransl"
        self.port = int(port)
        self.discovery_port = int(discovery_port)
        self.media_library = media_library
        if metadata_client is None and media_library is not None:
            # Construction is side-effect free; network access only happens after the
            # authenticated, explicit metadata-refresh endpoint is called.
            from dlsite_metadata import DlsiteMetadataClient
            metadata_client = DlsiteMetadataClient(media_library)
        self.metadata_client = metadata_client
        self._metadata_tasks: dict[str, dict[str, Any]] = {}
        self._metadata_tasks_lock = threading.RLock()
        self.allowed_networks = []
        for value in allowed_networks or ():
            try:
                self.allowed_networks.append(ipaddress.ip_network(str(value).strip(), strict=False))
            except ValueError:
                continue
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

    def metadata_task(self, task_id: str) -> dict[str, Any] | None:
        with self._metadata_tasks_lock:
            task = self._metadata_tasks.get(task_id)
            return json.loads(json.dumps(task, ensure_ascii=False)) if task else None

    def _metadata_target(self, work_id: str, product_id: str = "") -> tuple[str, str]:
        if self.media_library is None or self.metadata_client is None:
            raise RuntimeError("metadata service unavailable")
        work = self.media_library.work(work_id)
        if not work:
            candidate = str(product_id or work_id).strip().upper()
            if not re.fullmatch(r"(?:RJ|VJ|BJ)\d{6,10}", candidate):
                raise KeyError("work not found and no DLsite product id was supplied")
            canonical_id = self.media_library.ensure_metadata_work(candidate)
            work = self.media_library.work(canonical_id)
        canonical_id = str(work.get("id") or work_id)
        identifier = str(work.get("product_id") or product_id or canonical_id).strip().upper()
        if not re.fullmatch(r"(?:RJ|VJ|BJ)\d{6,10}", identifier):
            raise ValueError("work does not have a DLsite product id")
        return canonical_id, identifier

    def refresh_metadata(self, work_id: str, product_id: str = "") -> dict[str, Any]:
        canonical_id, identifier = self._metadata_target(work_id, product_id)
        task_id = uuid.uuid4().hex
        now = time.time()
        task = {
            "id": task_id,
            "work_id": canonical_id,
            "product_id": identifier,
            "state": "queued",
            "error": "",
            "total": 1,
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "current_work_id": canonical_id,
            "created_at": now,
            "updated_at": now,
        }
        with self._metadata_tasks_lock:
            self._metadata_tasks[task_id] = task

        def run() -> None:
            with self._metadata_tasks_lock:
                task["state"] = "running"
                task["updated_at"] = time.time()
            try:
                self.metadata_client.enrich(canonical_id, identifier, force=True)
                refreshed = self.media_library.work(canonical_id)
                with self._metadata_tasks_lock:
                    task["state"] = "succeeded"
                    task["completed"] = 1
                    task["succeeded"] = 1
                    task["work"] = refreshed
                    task["updated_at"] = time.time()
            except Exception as error:
                with self._metadata_tasks_lock:
                    task["state"] = "failed"
                    task["completed"] = 1
                    task["failed"] = 1
                    task["error"] = str(error)
                    task["updated_at"] = time.time()

        threading.Thread(
            target=run,
            name=f"voicetransl-metadata-{task_id[:8]}",
            daemon=True,
        ).start()
        return self.metadata_task(task_id) or task

    def refresh_metadata_batch(self, works: list[Any]) -> dict[str, Any]:
        if self.media_library is None or self.metadata_client is None:
            raise RuntimeError("metadata service unavailable")
        if not isinstance(works, list) or not works:
            raise ValueError("at least one work is required")
        if len(works) > 500:
            raise ValueError("metadata batch is limited to 500 works")
        requested: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for value in works:
            if isinstance(value, dict):
                work_id = str(value.get("id") or value.get("work_id") or "").strip()
                product_id = str(value.get("product_id") or "").strip().upper()
            else:
                work_id = str(value or "").strip()
                product_id = ""
            key = (work_id, product_id)
            if work_id and key not in seen:
                seen.add(key)
                requested.append(key)
        if not requested:
            raise ValueError("at least one valid work is required")

        task_id = uuid.uuid4().hex
        now = time.time()
        task = {
            "id": task_id,
            "work_id": "",
            "product_id": "",
            "state": "queued",
            "error": "",
            "total": len(requested),
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "current_work_id": "",
            "failures": [],
            "created_at": now,
            "updated_at": now,
        }
        with self._metadata_tasks_lock:
            self._metadata_tasks[task_id] = task

        def run() -> None:
            with self._metadata_tasks_lock:
                task["state"] = "running"
                task["updated_at"] = time.time()
            for requested_id, supplied_product_id in requested:
                with self._metadata_tasks_lock:
                    task["current_work_id"] = supplied_product_id or requested_id
                    task["updated_at"] = time.time()
                try:
                    canonical_id, identifier = self._metadata_target(requested_id, supplied_product_id)
                    self.metadata_client.enrich(canonical_id, identifier, force=True)
                    with self._metadata_tasks_lock:
                        task["succeeded"] += 1
                except Exception as error:
                    with self._metadata_tasks_lock:
                        task["failed"] += 1
                        task["failures"].append({
                            "work_id": requested_id,
                            "product_id": supplied_product_id,
                            "error": str(error),
                        })
                finally:
                    with self._metadata_tasks_lock:
                        task["completed"] += 1
                        task["updated_at"] = time.time()
            with self._metadata_tasks_lock:
                task["state"] = "succeeded"
                task["current_work_id"] = ""
                task["updated_at"] = time.time()

        threading.Thread(
            target=run,
            name=f"voicetransl-metadata-batch-{task_id[:8]}",
            daemon=True,
        ).start()
        return self.metadata_task(task_id) or task

    def _is_allowed_client(self, host: str) -> bool:
        if _is_private_client(host):
            return True
        try:
            address = ipaddress.ip_address(host.split("%", 1)[0])
        except ValueError:
            return False
        return any(address in network for network in self.allowed_networks)

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

    def _load_devices(self) -> dict[str, dict[str, Any]]:
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

    def paired_devices(self) -> list[dict[str, Any]]:
        with self._devices_lock:
            return [
                {
                    "id": key,
                    "name": value.get("name", key),
                    "file_management": bool(value.get("file_management", False)),
                }
                for key, value in self._devices.items()
            ]

    def set_file_management(self, device_id: str, enabled: bool) -> None:
        with self._devices_lock:
            if device_id not in self._devices:
                raise KeyError("paired device not found")
            self._devices[device_id]["file_management"] = bool(enabled)
            self._save_devices()

    def can_manage_files(self, device_id: str) -> bool:
        with self._devices_lock:
            return bool(self._devices.get(device_id, {}).get("file_management", False))

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
                if payload.strip() != DISCOVERY_REQUEST or not self._is_allowed_client(address[0]):
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
            previous = self._devices.get(device_id, {})
            self._devices[device_id] = {
                "name": device_name[:80],
                "token_hash": digest,
                "file_management": bool(previous.get("file_management", False)),
            }
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
                if service._is_allowed_client(self.client_address[0]):
                    return True
                self._json(HTTPStatus.FORBIDDEN, {"error": "LAN clients only"})
                return False

            def _auth(self) -> str | None:
                device_id, valid = service._authenticate(self.headers.get("Authorization", ""))
                if valid:
                    return device_id
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "invalid device token"})
                return None

            def _library(self) -> Any | None:
                if service.media_library is not None:
                    return service.media_library
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "media library unavailable"})
                return None

            def _require_file_management(self, device_id: str) -> bool:
                if service.can_manage_files(device_id):
                    return True
                self._json(HTTPStatus.FORBIDDEN, {"error": "file management is not enabled for this device"})
                return False

            def _serve_library_asset(self, asset_id: str, *, attachment: bool, send_body: bool = True) -> None:
                library = self._library()
                if library is None:
                    return
                asset = library.asset(asset_id)
                if not asset:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "asset not found"})
                    return
                path = Path(asset["path"])
                size = path.stat().st_size
                start, end = 0, max(0, size - 1)
                status = HTTPStatus.OK
                range_value = self.headers.get("Range", "").strip()
                if range_value:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_value)
                    if not match or (not match.group(1) and not match.group(2)):
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                    if match.group(1):
                        start = int(match.group(1))
                        end = int(match.group(2)) if match.group(2) else end
                    else:
                        suffix = int(match.group(2))
                        start = max(0, size - suffix)
                    if start >= size or start > end:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                    end = min(end, size - 1)
                    status = HTTPStatus.PARTIAL_CONTENT
                length = max(0, end - start + 1)
                self.send_response(status)
                self.send_header("Content-Type", asset.get("mime_type") or "application/octet-stream")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", f'"{asset["fingerprint"]}"')
                if status == HTTPStatus.PARTIAL_CONTENT:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header(
                    "Content-Disposition",
                    _content_disposition(path.name, attachment=attachment),
                )
                self.end_headers()
                if not send_body:
                    return
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

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
                if parts[:3] == ["api", "v1", "library"]:
                    library = self._library()
                    if library is None:
                        return
                    query = parse_qs(parsed.query, keep_blank_values=True)
                    try:
                        if parts == ["api", "v1", "library", "sync"]:
                            since = float(query.get("since", ["0"])[0] or 0)
                            limit = int(query.get("limit", ["500"])[0] or 500)
                            self._json(HTTPStatus.OK, library.sync(since=since, limit=limit))
                            return
                        if parts == ["api", "v1", "library", "search"]:
                            self._json(HTTPStatus.OK, library.search(
                                query.get("q", [""])[0],
                                tag=query.get("tag", [""])[0],
                                maker=query.get("maker", [""])[0],
                                series=query.get("series", [""])[0],
                                limit=int(query.get("limit", ["100"])[0] or 100),
                                offset=int(query.get("offset", ["0"])[0] or 0),
                            ))
                            return
                        if parts == ["api", "v1", "library", "inbox"]:
                            self._json(HTTPStatus.OK, {"items": library.inbox(query.get("status", ["open"])[0])})
                            return
                        if len(parts) == 6 and parts[:5] == ["api", "v1", "library", "organizer", "plans"]:
                            plan = library.organizer_plan(parts[5])
                            self._json(HTTPStatus.OK, plan) if plan else self._json(
                                HTTPStatus.NOT_FOUND, {"error": "organizer plan not found"}
                            )
                            return
                        if len(parts) == 5 and parts[:4] == ["api", "v1", "library", "metadata-refresh"]:
                            task = service.metadata_task(parts[4])
                            self._json(HTTPStatus.OK, task) if task else self._json(
                                HTTPStatus.NOT_FOUND, {"error": "metadata task not found"}
                            )
                            return
                        if len(parts) == 5 and parts[:4] == ["api", "v1", "library", "works"]:
                            work = library.work(parts[4])
                            self._json(HTTPStatus.OK, work) if work else self._json(
                                HTTPStatus.NOT_FOUND, {"error": "work not found"}
                            )
                            return
                        if len(parts) == 6 and parts[:4] == ["api", "v1", "library", "assets"]:
                            if parts[5] in {"stream", "download"}:
                                self._serve_library_asset(parts[4], attachment=parts[5] == "download")
                                return
                    except (TypeError, ValueError) as error:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
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
                    self.send_header(
                        "Content-Disposition",
                        _content_disposition(path.name, attachment=True),
                    )
                    self.end_headers()
                    with path.open("rb") as stream:
                        shutil.copyfileobj(stream, self.wfile, 1024 * 1024)
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_HEAD(self) -> None:  # noqa: N802
                if not self._private():
                    return
                if self._auth() is None:
                    return
                parsed = urlparse(self.path)
                parts = [unquote(x) for x in parsed.path.strip("/").split("/") if x]
                if len(parts) == 6 and parts[:4] == ["api", "v1", "library", "assets"] and parts[5] in {"stream", "download"}:
                    self._serve_library_asset(parts[4], attachment=parts[5] == "download", send_body=False)
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
                if parts == ["api", "v1", "library", "organizer", "preview"]:
                    try:
                        body = self._body_json(maximum=256 * 1024)
                        library = self._library()
                        if library is None:
                            return
                        plan = library.organizer_preview(
                            asset_ids=body.get("asset_ids", []),
                            work_ids=body.get("work_ids", []),
                            action=str(body.get("action", "organize")),
                            destination=str(body.get("destination", "")),
                            new_name=str(body.get("new_name", "")),
                            device_id=device_id,
                        )
                        self._json(HTTPStatus.CREATED, plan)
                    except (OSError, TypeError, ValueError) as error:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                if len(parts) == 7 and parts[:5] == ["api", "v1", "library", "organizer", "plans"] and parts[6] == "apply":
                    if not self._require_file_management(device_id):
                        return
                    library = self._library()
                    if library is None:
                        return
                    try:
                        self._json(HTTPStatus.OK, library.apply_organizer_plan(unquote(parts[5])))
                    except KeyError as error:
                        self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                    except (OSError, ValueError) as error:
                        self._json(HTTPStatus.CONFLICT, {"error": str(error)})
                    return
                if len(parts) == 7 and parts[:5] == ["api", "v1", "library", "organizer", "operations"] and parts[6] == "undo":
                    if not self._require_file_management(device_id):
                        return
                    library = self._library()
                    if library is None:
                        return
                    try:
                        self._json(HTTPStatus.OK, library.undo_organizer_operation(unquote(parts[5])))
                    except KeyError as error:
                        self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                    except (OSError, ValueError) as error:
                        self._json(HTTPStatus.CONFLICT, {"error": str(error)})
                    return
                if len(parts) == 6 and parts[:4] == ["api", "v1", "library", "assets"] and parts[5] == "reassign":
                    if not self._require_file_management(device_id):
                        return
                    library = self._library()
                    if library is None:
                        return
                    try:
                        body = self._body_json()
                        self._json(HTTPStatus.OK, library.reassign_asset(unquote(parts[4]), str(body.get("work_id", ""))))
                    except KeyError as error:
                        self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                    except ValueError as error:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                if len(parts) == 6 and parts[:4] == ["api", "v1", "library", "works"] and parts[5] == "cover":
                    if not self._require_file_management(device_id):
                        return
                    library = self._library()
                    if library is None:
                        return
                    try:
                        body = self._body_json()
                        self._json(HTTPStatus.OK, library.set_manual_cover(unquote(parts[4]), str(body.get("asset_id", ""))))
                    except KeyError as error:
                        self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                    except ValueError as error:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                if parts == ["api", "v1", "library", "metadata-refresh"]:
                    try:
                        body = self._body_json(maximum=128 * 1024)
                        works = body.get("works", body.get("work_ids", []))
                        task = service.refresh_metadata_batch(works)
                        self._json(HTTPStatus.ACCEPTED, task)
                    except (RuntimeError, TypeError, ValueError) as error:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                if len(parts) == 6 and parts[:4] == ["api", "v1", "library", "works"] and parts[5] == "metadata-refresh":
                    try:
                        body = self._body_json() if int(self.headers.get("Content-Length", "0") or 0) else {}
                        task = service.refresh_metadata(
                            unquote(parts[4]),
                            str(body.get("product_id") or ""),
                        )
                        self._json(HTTPStatus.ACCEPTED, task)
                    except KeyError as error:
                        self._json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                    except (RuntimeError, ValueError) as error:
                        self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
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
