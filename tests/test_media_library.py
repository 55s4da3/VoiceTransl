import json
import tempfile
import unittest
from pathlib import Path

from media_library import (
    MediaLibraryDatabase,
    MediaLibraryScanner,
    normalize_product_id,
    normalized_title,
    subtitle_language,
    write_preview,
)


class MediaLibraryScannerTest(unittest.TestCase):
    def test_tg_subtitle_is_classified_as_chinese_translation(self):
        self.assertEqual("zh", subtitle_language(Path("chapter.resegmented.tg.srt")))

    def test_product_ids_and_title_normalization(self):
        self.assertEqual("RJ01392175", normalize_product_id("[rj01392175][MP3] title"))
        self.assertEqual("title", normalized_title("[RJ01392175][MP3] Title"))

    def test_scan_is_read_only_and_attaches_translated_media_and_subtitles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "media"
            work = root / "RJ01392175 Sample work"
            translated = root / "已翻译" / "RJ01392175"
            ordinary = root / "Ordinary voice collection"
            work.mkdir(parents=True)
            translated.mkdir(parents=True)
            ordinary.mkdir(parents=True)
            audio = work / "01_intro.mp3"
            subtitle = translated / "01_intro.zh-cn.srt"
            translated_audio = translated / "02_translated_copy.m4a"
            translated_image = translated / "cover.jpg"
            audio.write_bytes(b"audio")
            subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
            translated_audio.write_bytes(b"translated voice")
            translated_image.write_bytes(b"image")
            (ordinary / "chapter one.m4a").write_bytes(b"voice")
            before = {
                str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in root.rglob("*") if path.is_file()
            }

            preview = MediaLibraryScanner().scan(root)

            after = {
                str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in root.rglob("*") if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual(2, len(preview.works))
            official = next(item for item in preview.works if item.product_id == "RJ01392175")
            self.assertEqual("auto", official.classification)
            self.assertEqual(4, len(official.assets))
            self.assertTrue(next(item for item in official.assets if item.kind == "subtitle").derived)
            self.assertFalse(
                next(item for item in official.assets if item.path == str(translated_audio.resolve())).derived
            )
            self.assertEqual("cover", next(item for item in official.assets if item.kind == "image").role)
            self.assertEqual(1, preview.summary["subtitles"])

    def test_translated_directory_can_be_the_only_source_of_playable_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            translated = root / "已翻译" / "RJ01546796"
            translated.mkdir(parents=True)
            audio = translated / "RJ01546796 - 01.m4a"
            subtitle = translated / "RJ01546796 - 01.srt"
            audio.write_bytes(b"audio")
            subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

            preview = MediaLibraryScanner().scan(root)

            self.assertEqual(1, len(preview.works))
            work = preview.works[0]
            self.assertEqual("RJ01546796", work.product_id)
            self.assertEqual("auto", work.classification)
            self.assertEqual({"audio", "subtitle"}, {asset.kind for asset in work.assets})
            self.assertFalse(next(asset for asset in work.assets if asset.kind == "audio").derived)
            self.assertTrue(next(asset for asset in work.assets if asset.kind == "subtitle").derived)

    def test_multiple_top_level_tracks_share_one_official_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "RJ01311315.mp3").write_bytes(b"main")
            (root / "RJ01311315 omake.mp3").write_bytes(b"bonus")
            preview = MediaLibraryScanner().scan(root)
            self.assertEqual(1, len(preview.works))
            self.assertEqual("auto", preview.works[0].classification)
            self.assertEqual(2, len(preview.works[0].assets))
            self.assertEqual("bonus", next(asset for asset in preview.works[0].assets if "omake" in asset.path).section)

    def test_nested_collections_use_the_nearest_product_id_and_classify_companions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "新建文件夹 (3)" / "[RJ230006] outer" / "[RJ387435] inner"
            nested.mkdir(parents=True)
            video = nested / "01 main.mp4"
            extra = nested / "02_エクストラ.mp4"
            image = nested / "イラスト" / "スチル.png"
            image.parent.mkdir()
            video.write_bytes(b"video")
            extra.write_bytes(b"extra")
            image.write_bytes(b"image")

            preview = MediaLibraryScanner().scan(root)

            work = next(item for item in preview.works if item.product_id == "RJ387435")
            self.assertEqual(3, len(work.assets))
            self.assertEqual("bonus", next(item for item in work.assets if item.path == str(extra.resolve())).section)
            self.assertEqual("illustration", next(item for item in work.assets if item.kind == "image").role)
            self.assertFalse(any(item.product_id == "RJ230006" for item in preview.works))

    def test_generic_collection_does_not_become_a_fake_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            creator = root / "fc" / "声優A"
            loose = root / "FYZ"
            creator.mkdir(parents=True)
            loose.mkdir(parents=True)
            (creator / "episode 1.mp3").write_bytes(b"audio")
            (creator / "episode 2.mp3").write_bytes(b"audio")
            (loose / "stream 2025.mp4").write_bytes(b"video")
            (loose / "stream 2025.combine.srt").write_text("subtitle", encoding="utf-8")

            preview = MediaLibraryScanner().scan(root)

            self.assertFalse(any(work.title.casefold() in {"fc", "fyz"} for work in preview.works))
            creator_work = next(work for work in preview.works if work.title == "声優A")
            self.assertEqual(2, len(creator_work.assets))
            stream_work = next(work for work in preview.works if work.title == "stream 2025")
            self.assertEqual({"video", "subtitle"}, {asset.kind for asset in stream_work.assets})

    def test_unique_known_title_attaches_orphan_artwork(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artwork = root / "イラスト"
            artwork.mkdir()
            (artwork / "メスイキ調教されるあなた_ロゴあり.png").write_bytes(b"image")
            preview = MediaLibraryScanner(
                known_titles={"RJ249851": "【TS百合】先輩とお姉さまにメスイキ調教されるあなた【バイノーラル】"}
            ).scan(root)
            self.assertEqual("RJ249851", preview.works[0].product_id)
            self.assertIn("unique_title_match", preview.works[0].reasons)


class MediaLibraryDatabaseTest(unittest.TestCase):
    def test_metadata_only_work_can_be_created_for_a_phone_local_product(self):
        with tempfile.TemporaryDirectory() as directory:
            database = MediaLibraryDatabase(Path(directory) / "catalogue.sqlite3")

            work_id = database.ensure_metadata_work("rj01316235")

            self.assertEqual("RJ01316235", work_id)
            work = database.work(work_id)
            self.assertEqual("RJ01316235", work["product_id"])
            self.assertEqual([], work["assets"])
            self.assertEqual(
                [{"id": "RJ01316235", "product_id": "RJ01316235"}],
                database.metadata_candidates(),
            )

    def test_apply_sync_search_metadata_and_asset_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media"
            (media / "RJ01010136").mkdir(parents=True)
            audio = media / "RJ01010136" / "01 sample.wav"
            audio.write_bytes(b"RIFF-test")
            (media / "One local work").mkdir()
            (media / "One local work" / "voice.mp3").write_bytes(b"local")
            (media / "Another local work").mkdir()
            (media / "Another local work" / "voice.mp3").write_bytes(b"local2")
            preview = MediaLibraryScanner().scan(media)
            database = MediaLibraryDatabase(root / "catalogue.sqlite3")
            scan_id = database.apply_preview(preview)
            self.assertTrue(scan_id)
            synced = database.sync()["works"]
            self.assertEqual(3, len(synced))
            self.assertTrue(any(work.get("assets") for work in synced))

            database.update_metadata(
                "RJ01010136",
                {
                    "title_ja": "日本語タイトル",
                    "title_zh": "中文标题",
                    "description_ja": "説明",
                    "description_zh": "简介",
                    "maker": "Circle",
                    "series": "Series",
                    "rating": 4.8,
                    "rating_count": 100,
                    "tags": ["ASMR", "耳かき"],
                    "people": {"voice_actor": ["Actor"]},
                    "source": "dlsite",
                },
            )
            result = database.search("中文")
            self.assertEqual(1, len(result["works"]))
            work = database.work("RJ01010136")
            self.assertEqual("中文标题", work["title"])
            self.assertEqual(["Actor"], work["people"]["voice_actor"])
            self.assertIn("ASMR", work["tags"])
            asset = database.asset(work["assets"][0]["id"])
            self.assertEqual(str(audio.resolve()), asset["path"])

    def test_preview_json_is_written_outside_media_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media"
            media.mkdir()
            (media / "RJ01010136.mp3").write_bytes(b"audio")
            preview = MediaLibraryScanner().scan(media)
            destination = write_preview(preview, root / "preview.json", include_assets=False)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(1, payload["summary"]["works"])
            self.assertNotIn("assets", payload["works"][0])

    def test_organizer_preview_apply_and_undo_preserve_asset_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media"
            source = media / "mixed" / "RJ01370011 - 09_特典.m4a"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"bonus")
            database = MediaLibraryDatabase(root / "catalogue.sqlite3")
            database.apply_preview(MediaLibraryScanner().scan(media))
            before = database.work("RJ01370011")["assets"][0]

            plan = database.organizer_preview(asset_ids=[before["id"]])
            self.assertEqual(1, plan["summary"]["ready"])
            operation = database.apply_organizer_plan(plan["id"])
            moved = database.asset(before["id"])
            self.assertEqual(before["id"], moved["id"])
            self.assertIn("特典", moved["path"])
            self.assertFalse(source.exists())

            result = database.undo_organizer_operation(operation["id"])
            self.assertEqual("undone", result["state"])
            self.assertTrue(source.exists())
            self.assertEqual(before["id"], database.asset(before["id"])["id"])

    def test_organizer_rejects_stale_or_conflicting_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media"
            source = media / "RJ01010136.mp3"
            media.mkdir()
            source.write_bytes(b"audio")
            database = MediaLibraryDatabase(root / "catalogue.sqlite3")
            database.apply_preview(MediaLibraryScanner().scan(media))
            asset = database.work("RJ01010136")["assets"][0]
            plan = database.organizer_preview(asset_ids=[asset["id"]], action="rename", new_name="renamed")
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                database.apply_organizer_plan(plan["id"])


if __name__ == "__main__":
    unittest.main()
