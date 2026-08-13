import tempfile
import unittest
from pathlib import Path

import requests

from lan_service import LanService, build_artifact


def _profile():
    public = {
        "revision": "abc123",
        "ready": True,
        "source_language": "ja",
        "target_language": "zh-cn",
    }
    return {"public": public, "snapshot": {"enable_translation": True}}


def _test_pair_upload_progress_artifact_and_cancel(tmp_path):
    ready = []
    cancelled = []
    service = LanService(
        root=tmp_path / "jobs",
        state_dir=tmp_path / "state",
        profile_provider=_profile,
        job_ready=ready.append,
        cancel_job=cancelled.append,
        port=0,
        discovery_port=0,
    )
    service.start()
    base = f"http://127.0.0.1:{service.port}"
    try:
        info = requests.get(f"{base}/api/v1/info", timeout=2)
        assert info.status_code == 200
        assert info.json()["api_version"] == 1

        paired = requests.post(
            f"{base}/api/v1/pair",
            json={"code": service.pair_code, "device_id": "phone-1", "device_name": "PixelPlayer"},
            timeout=2,
        )
        assert paired.status_code == 200
        headers = {"Authorization": f"Bearer {paired.json()['token']}"}
        assert requests.get(f"{base}/api/v1/profile", headers=headers, timeout=2).json()["revision"] == "abc123"

        payload = b"fake audio bytes"
        created = requests.post(
            f"{base}/api/v1/jobs",
            headers=headers,
            json={"filename": "../sample.wav", "size": len(payload), "profile_revision": "abc123"},
            timeout=2,
        )
        assert created.status_code == 201
        job_id = created.json()["id"]
        assert created.json()["filename"] == "sample.wav"
        uploaded = requests.put(
            f"{base}/api/v1/jobs/{job_id}/audio",
            headers=headers,
            data=payload,
            timeout=2,
        )
        assert uploaded.status_code == 200
        assert ready == [job_id]
        input_path, output_dir = service.registry.paths(job_id)
        assert input_path.read_bytes() == payload

        subtitle = output_dir / "sample.combine.srt"
        subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n你好\n", encoding="utf-8")
        service.registry.set_artifacts(job_id, [build_artifact(subtitle, "combined")])
        status = requests.get(f"{base}/api/v1/jobs/{job_id}", headers=headers, timeout=2).json()
        assert status["state"] == "succeeded"
        downloaded = requests.get(
            f"{base}/api/v1/jobs/{job_id}/artifacts/combined", headers=headers, timeout=2
        )
        assert downloaded.status_code == 200
        assert downloaded.content == subtitle.read_bytes()

        response = requests.delete(f"{base}/api/v1/jobs/{job_id}", headers=headers, timeout=2)
        assert response.status_code == 202
        assert cancelled == [job_id]
        assert requests.get(f"{base}/api/v1/profile", timeout=2).status_code == 401
    finally:
        service.stop()


def _test_profile_revision_and_device_isolation(tmp_path):
    service = LanService(
        root=tmp_path / "jobs",
        state_dir=tmp_path / "state",
        profile_provider=_profile,
        job_ready=lambda _job: None,
        cancel_job=lambda _job: None,
        port=0,
        discovery_port=0,
    )
    service.start()
    base = f"http://127.0.0.1:{service.port}"
    try:
        paired = requests.post(
            f"{base}/api/v1/pair",
            json={"code": service.pair_code, "device_id": "phone", "device_name": "Phone"},
            timeout=2,
        ).json()
        headers = {"Authorization": f"Bearer {paired['token']}"}
        conflict = requests.post(
            f"{base}/api/v1/jobs",
            headers=headers,
            json={"filename": "x.wav", "size": 1, "profile_revision": "old"},
            timeout=2,
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"] == "profile_changed"
        assert requests.get(
            f"{base}/api/v1/jobs/not-a-job", headers=headers, timeout=2
        ).status_code == 404
    finally:
        service.stop()


class LanServiceTest(unittest.TestCase):
    def test_pair_upload_progress_artifact_and_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            _test_pair_upload_progress_artifact_and_cancel(Path(directory))

    def test_profile_revision_and_device_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            _test_profile_revision_and_device_isolation(Path(directory))

    def test_restart_marks_incomplete_jobs_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_service = LanService(
                root=root / "jobs",
                state_dir=root / "state",
                profile_provider=_profile,
                job_ready=lambda _job: None,
                cancel_job=lambda _job: None,
                port=0,
                discovery_port=0,
            )
            job = registry_service.registry.create(
                device_id="phone",
                filename="audio.wav",
                size=10,
                profile=_profile()["public"],
                snapshot=_profile()["snapshot"],
            )
            registry_service.registry.update(job["id"], state="running")

            restarted = LanService(
                root=root / "jobs",
                state_dir=root / "state",
                profile_provider=_profile,
                job_ready=lambda _job: None,
                cancel_job=lambda _job: None,
                port=0,
                discovery_port=0,
            )
            recovered = restarted.registry.public(job["id"])
            self.assertIsNotNone(recovered)
            self.assertEqual("failed", recovered["state"])
            self.assertIn("restarted", recovered["error"])
