import tempfile
import time
import unittest
from pathlib import Path

import requests

from lan_service import LanService, build_artifact
from media_library import MediaLibraryDatabase, MediaLibraryScanner


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

    def test_media_library_sync_search_details_and_range_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media" / "RJ01630025"
            media.mkdir(parents=True)
            audio = media / "01_日本語.mp3"
            audio.write_bytes(b"0123456789abcdef")
            library = MediaLibraryDatabase(root / "library.sqlite3")
            library.apply_preview(MediaLibraryScanner().scan(root / "media"))
            library.update_metadata(
                "RJ01630025",
                {"title_zh": "中文作品", "title_ja": "日本語作品", "tags": ["ASMR"]},
            )
            class FakeMetadataClient:
                def __init__(self):
                    self.calls = []

                def enrich(self, work_id, product_id, *, force=False):
                    self.calls.append((work_id, product_id, force))
                    value = {"title_zh": "refreshed", "title_ja": "refreshed", "tags": ["ASMR"]}
                    library.update_metadata(work_id, value)
                    return value

            metadata_client = FakeMetadataClient()
            service = LanService(
                root=root / "jobs",
                state_dir=root / "state",
                profile_provider=_profile,
                job_ready=lambda _job: None,
                cancel_job=lambda _job: None,
                media_library=library,
                metadata_client=metadata_client,
                allowed_networks=["8.8.8.8/32", "invalid"],
                port=0,
                discovery_port=0,
            )
            self.assertTrue(service._is_allowed_client("8.8.8.8"))
            service.start()
            base = f"http://127.0.0.1:{service.port}"
            try:
                paired = requests.post(
                    f"{base}/api/v1/pair",
                    json={"code": service.pair_code, "device_id": "phone", "device_name": "Phone"},
                    timeout=2,
                ).json()
                headers = {"Authorization": f"Bearer {paired['token']}"}
                sync = requests.get(f"{base}/api/v1/library/sync", headers=headers, timeout=2)
                assert sync.status_code == 200
                assert sync.json()["works"][0]["assets"]
                assert sync.json()["works"][0]["title"] == "中文作品"
                synced_asset = sync.json()["works"][0]["assets"][0]
                assert synced_asset["section"] == "main"
                assert synced_asset["role"] == "chapter_audio"
                assert "group_key" in synced_asset
                search = requests.get(
                    f"{base}/api/v1/library/search", headers=headers, params={"q": "中文"}, timeout=2
                ).json()
                assert len(search["works"]) == 1
                details = requests.get(
                    f"{base}/api/v1/library/works/RJ01630025", headers=headers, timeout=2
                ).json()
                asset_id = next(item["id"] for item in details["assets"] if item["kind"] == "audio")
                ranged = requests.get(
                    f"{base}/api/v1/library/assets/{asset_id}/stream",
                    headers={**headers, "Range": "bytes=2-6"},
                    timeout=2,
                )
                assert ranged.status_code == 206
                assert ranged.content == b"23456"
                assert ranged.headers["Content-Range"] == "bytes 2-6/16"
                assert "filename*=UTF-8''" in ranged.headers["Content-Disposition"]
                headed = requests.head(
                    f"{base}/api/v1/library/assets/{asset_id}/download", headers=headers, timeout=2
                )
                assert headed.status_code == 200
                assert headed.headers["Content-Length"] == "16"
                assert requests.get(f"{base}/api/v1/library/sync", timeout=2).status_code == 401
                preview = requests.post(
                    f"{base}/api/v1/library/organizer/preview",
                    headers=headers,
                    json={"asset_ids": [asset_id], "action": "rename", "new_name": "renamed"},
                    timeout=2,
                )
                assert preview.status_code == 201
                plan_id = preview.json()["id"]
                assert requests.get(
                    f"{base}/api/v1/library/organizer/plans/{plan_id}", headers=headers, timeout=2
                ).status_code == 200
                denied = requests.post(
                    f"{base}/api/v1/library/organizer/plans/{plan_id}/apply", headers=headers, timeout=2
                )
                assert denied.status_code == 403
                service.set_file_management("phone", True)
                applied = requests.post(
                    f"{base}/api/v1/library/organizer/plans/{plan_id}/apply", headers=headers, timeout=2
                )
                assert applied.status_code == 200
                operation_id = applied.json()["id"]
                assert requests.post(
                    f"{base}/api/v1/library/organizer/operations/{operation_id}/undo",
                    headers=headers,
                    timeout=2,
                ).status_code == 200
                assert audio.exists()
                refresh = requests.post(
                    f"{base}/api/v1/library/works/RJ01630025/metadata-refresh",
                    headers=headers,
                    timeout=2,
                )
                assert refresh.status_code == 202
                task_id = refresh.json()["id"]
                deadline = time.time() + 2
                while time.time() < deadline:
                    task = requests.get(
                        f"{base}/api/v1/library/metadata-refresh/{task_id}", headers=headers, timeout=2
                    ).json()
                    if task["state"] in {"succeeded", "failed"}:
                        break
                    time.sleep(0.01)
                assert task["state"] == "succeeded"
                assert metadata_client.calls[-1] == ("RJ01630025", "RJ01630025", True)

                missing = requests.post(
                    f"{base}/api/v1/library/works/RJ01316235/metadata-refresh",
                    headers=headers,
                    json={"product_id": "RJ01316235"},
                    timeout=2,
                )
                assert missing.status_code == 202
                task_id = missing.json()["id"]
                deadline = time.time() + 2
                while time.time() < deadline:
                    task = requests.get(
                        f"{base}/api/v1/library/metadata-refresh/{task_id}", headers=headers, timeout=2
                    ).json()
                    if task["state"] in {"succeeded", "failed"}:
                        break
                    time.sleep(0.01)
                assert task["state"] == "succeeded"
                assert library.work("RJ01316235")["title"] == "refreshed"

                batch = requests.post(
                    f"{base}/api/v1/library/metadata-refresh",
                    headers=headers,
                    json={
                        "works": [
                            {"id": "RJ01630025", "product_id": "RJ01630025"},
                            {"id": "RJ01316235", "product_id": "RJ01316235"},
                        ]
                    },
                    timeout=2,
                )
                assert batch.status_code == 202
                task_id = batch.json()["id"]
                deadline = time.time() + 2
                while time.time() < deadline:
                    task = requests.get(
                        f"{base}/api/v1/library/metadata-refresh/{task_id}", headers=headers, timeout=2
                    ).json()
                    if task["state"] in {"succeeded", "failed"}:
                        break
                    time.sleep(0.01)
                assert task["state"] == "succeeded"
                assert task["total"] == 2
                assert task["completed"] == 2
                assert task["succeeded"] == 2
                assert task["failed"] == 0
            finally:
                service.stop()
