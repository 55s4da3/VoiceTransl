import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError

from dlsite_metadata import DlsiteMetadataClient, parse_dlsite_page
from media_library import MediaLibraryDatabase, MediaLibraryScanner


SAMPLE_JA = """
<html><head>
<meta property="og:image" content="https://img.example/cover.jpg">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product","name":"日本語作品",
 "description":"<p>日本語の説明</p>","brand":{"name":"サークルA"},
 "image":["https://img.example/cover.jpg"],"releaseDate":"2026-01-02",
 "aggregateRating":{"ratingValue":"4.8","ratingCount":"123"}}
</script></head><body>
<table><tr><th>シリーズ名</th><td><a>シリーズX</a></td></tr>
<tr><th>声優</th><td><a>声優A</a><a>声優B</a></td></tr>
<tr><th>シナリオ</th><td><a>作家A</a></td></tr>
<tr><th>ジャンル</th><td><a>ASMR</a><a>耳かき</a></td></tr></table>
</body></html>
"""

SAMPLE_ZH = """
<html><head><script type="application/ld+json">
{"@type":"Product","name":"中文作品","description":"<p>中文简介</p>",
 "brand":{"name":"社团A"},"aggregateRating":{"ratingValue":4.8,"ratingCount":123}}
</script></head><body><table>
<tr><th>声优</th><td><a>声优A</a></td></tr>
<tr><th>分类</th><td><a>ASMR</a><a>治愈</a></td></tr>
</table></body></html>
"""

SAMPLE_CURRENT_PAGE = """
<html><head>
<meta property="og:title" content="真实作品标题 [测试社团] | DLsite">
<meta property="og:description" content="真正的作品简介。『DLsite 同人 - R18』は同人誌・同人ゲームのダウンロードショップ。">
<meta property="og:image" content="https://img.example/current.jpg">
<meta name="rating" content="adult">
<script type="application/ld+json">
{"@type":"BreadcrumbList","itemListElement":[
 {"@type":"ListItem","name":"同人"},
 {"@type":"ListItem","name":"真实作品标题"}]}
</script>
</head><body>
<h1 id="work_name">真实作品标题</h1>
<div itemprop="aggregateRating">
 <meta itemprop="ratingValue" content="4.8">
 <meta content="271" itemprop="ratingCount">
</div>
<table><tr><th>サークル名</th><td><a>测试社团</a></td></tr></table>
</body></html>
"""


class _Response:
    def __init__(self, payload):
        self.status = 200
        self.headers = {"ETag": "fixture"}
        self._payload = payload

    def read(self):
        return self._payload


class DlsiteMetadataTest(unittest.TestCase):
    def test_parse_json_ld_and_outline_fields(self):
        value = parse_dlsite_page(SAMPLE_JA, locale="ja", product_id="RJ01630025")
        self.assertEqual("日本語作品", value["title_ja"])
        self.assertEqual("日本語の説明", value["description_ja"])
        self.assertEqual("サークルA", value["maker"])
        self.assertEqual("シリーズX", value["series"])
        self.assertEqual(4.8, value["rating"])
        self.assertEqual(123, value["rating_count"])
        self.assertEqual(["声優A", "声優B"], value["people"]["voice_actor"])
        self.assertEqual(["ASMR", "耳かき"], value["tags"])

    def test_parse_current_dlsite_markup_without_product_json_ld(self):
        value = parse_dlsite_page(SAMPLE_CURRENT_PAGE, locale="ja", product_id="RJ01114383")
        self.assertEqual("真实作品标题", value["title_ja"])
        self.assertEqual("真正的作品简介。", value["description_ja"])
        self.assertEqual("测试社团", value["maker"])
        self.assertEqual(4.8, value["rating"])
        self.assertEqual(271, value["rating_count"])
        self.assertEqual("adult", value["age_rating"])

    def test_untranslated_chinese_page_does_not_replace_japanese_title(self):
        with tempfile.TemporaryDirectory() as directory:
            database = MediaLibraryDatabase(Path(directory) / "library.sqlite3")

            def opener(request, timeout):
                return _Response(SAMPLE_CURRENT_PAGE.encode("utf-8"))

            value = DlsiteMetadataClient(database, min_interval=0, opener=opener).fetch("RJ01114383")
            self.assertEqual("真实作品标题", value["title_ja"])
            self.assertIn("title_zh", value)
            self.assertEqual("", value["title_zh"])
            self.assertEqual("真正的作品简介。", value["description_ja"])
            self.assertEqual("", value["description_zh"])

    def test_fetch_merges_locales_and_uses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            database = MediaLibraryDatabase(Path(directory) / "library.sqlite3")
            calls = []

            def opener(request, timeout):
                calls.append(request.full_url)
                return _Response((SAMPLE_ZH if "zh_CN" in request.full_url else SAMPLE_JA).encode("utf-8"))

            client = DlsiteMetadataClient(database, min_interval=0, opener=opener)
            first = client.fetch("RJ01630025")
            second = client.fetch("RJ01630025")
            self.assertEqual(2, len(calls))
            self.assertEqual(first, second)
            self.assertEqual("日本語作品", first["title_ja"])
            self.assertEqual("中文作品", first["title_zh"])
            self.assertEqual(["ASMR", "耳かき", "治愈"], first["tags"])

    def test_force_refresh_falls_back_to_last_successful_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            database = MediaLibraryDatabase(Path(directory) / "library.sqlite3")

            def opener(request, timeout):
                return _Response((SAMPLE_ZH if "zh_CN" in request.full_url else SAMPLE_JA).encode("utf-8"))

            client = DlsiteMetadataClient(database, min_interval=0, opener=opener)
            expected = client.fetch("RJ01630025")

            def offline(_request, timeout):
                raise URLError("timed out")

            client.opener = offline
            actual = client.fetch("RJ01630025", force=True)
            self.assertEqual(expected, actual)

    def test_enrich_preserves_japanese_and_prefers_chinese_title(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media" / "RJ01630025"
            media.mkdir(parents=True)
            (media / "01.mp3").write_bytes(b"audio")
            database = MediaLibraryDatabase(root / "library.sqlite3")
            database.apply_preview(MediaLibraryScanner().scan(root / "media"))

            def opener(request, timeout):
                return _Response((SAMPLE_ZH if "zh_CN" in request.full_url else SAMPLE_JA).encode("utf-8"))

            DlsiteMetadataClient(database, min_interval=0, opener=opener).enrich(
                "RJ01630025", "RJ01630025"
            )
            work = database.work("RJ01630025")
            self.assertEqual("中文作品", work["title"])
            self.assertEqual("日本語作品", work["title_ja"])
            self.assertEqual("中文作品", work["title_zh"])

    def test_enrich_can_clear_stale_localised_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media" / "RJ01114383"
            media.mkdir(parents=True)
            (media / "01.mp3").write_bytes(b"audio")
            database = MediaLibraryDatabase(root / "library.sqlite3")
            database.apply_preview(MediaLibraryScanner().scan(root / "media"))
            database.update_metadata(
                "RJ01114383",
                {
                    "source": "dlsite",
                    "title_ja": "错误日文",
                    "title_zh": "错误中文",
                    "description_zh": "错误简介",
                },
            )

            def opener(request, timeout):
                return _Response(SAMPLE_CURRENT_PAGE.encode("utf-8"))

            DlsiteMetadataClient(database, min_interval=0, opener=opener).enrich(
                "RJ01114383", "RJ01114383"
            )
            work = database.work("RJ01114383")
            self.assertEqual("真实作品标题", work["title"])
            self.assertEqual("真实作品标题", work["title_ja"])
            self.assertEqual("", work["title_zh"])
            self.assertEqual("", work["description_zh"])


if __name__ == "__main__":
    unittest.main()
