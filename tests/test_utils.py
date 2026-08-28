import lzma
import tempfile
import unittest
from pathlib import Path

from GalTransl.Backend.GenDic import (
    _extract_regex_terms,
    _is_katakana_only,
    solve_sentence_selection,
)
from GalTransl.Utils import (
    contains_english,
    contains_japanese,
    contains_katakana,
    contains_korean,
    extract_code_blocks,
    extract_control_substrings,
    find_most_repeated_substring,
    fix_quotes2,
    get_file_list,
    get_file_name,
    get_most_common_char,
    get_n_symbol,
    is_all_chinese,
    is_all_gbk,
    process_escape,
    decompress_file_lzma,
)


class TextUtilityTests(unittest.TestCase):
    def test_extracts_control_substrings(self):
        self.assertEqual(
            extract_control_substrings(r"前文\n[var_1] 后文 @tag-2"),
            [r"\n[var_1]", "@tag-2"],
        )

    def test_most_common_character_ignores_blacklist(self):
        self.assertEqual(get_most_common_char("....，，甲甲乙"), ("甲", 2))

    def test_language_detectors(self):
        self.assertEqual(set(contains_japanese("中文かなカナ英")), set("かなカナ"))
        self.assertTrue(contains_korean("中文한글"))
        self.assertFalse(contains_korean("中文"))
        self.assertTrue(contains_katakana("ゲーム"))
        self.assertFalse(contains_katakana("かなー"))
        self.assertEqual(contains_english("中AbＣｄ文"), "AbＣｄ")

    def test_all_chinese_requires_nonempty_chinese_only(self):
        self.assertTrue(is_all_chinese("中文漢字"))
        self.assertFalse(is_all_chinese("中文A"))
        self.assertFalse(is_all_chinese(""))

    def test_non_gbk_characters_are_returned(self):
        self.assertEqual(is_all_gbk("中文😀"), "😀")
        self.assertEqual(is_all_gbk("中文"), "")
        self.assertEqual(is_all_gbk(""), "")

    def test_extract_code_blocks_with_languages(self):
        langs, code = extract_code_blocks("```py\nprint(1)\n```\n```\ntext\n```")
        self.assertEqual(langs, ["py", ""])
        self.assertEqual(code, ["print(1)", "text"])

    def test_path_and_escape_helpers(self):
        self.assertEqual(get_file_name("folder/archive.tar.gz"), "archive.tar")
        self.assertEqual(process_escape(r"line\nnext"), "line\nnext")
        self.assertEqual(get_n_symbol("a\\r\\nb"), [r"\r\n"])
        self.assertEqual(get_n_symbol("a\nb"), ["\n"])

    def test_quote_normalization(self):
        self.assertEqual(fix_quotes2('"你好"'), "“你好”")
        self.assertEqual(fix_quotes2('他说"好"'), "他说“好”")

    def test_repeated_substring(self):
        substring, count = find_most_repeated_substring("abcabcabcx")
        self.assertEqual((substring, count), ("abc", 3))

    def test_recursive_file_listing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "nested").mkdir()
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "nested" / "b.txt").write_text("b", encoding="utf-8")
            self.assertEqual(
                {Path(path).name for path in get_file_list(temp_dir)},
                {"a.txt", "b.txt"},
            )

    def test_lzma_decompression_with_default_and_explicit_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "sample.bin.xz"
            source.write_bytes(lzma.compress(b"payload"))
            decompress_file_lzma(str(source))
            self.assertEqual((root / "sample.bin").read_bytes(), b"payload")

            explicit = root / "copy.bin"
            decompress_file_lzma(str(source), str(explicit))
            self.assertEqual(explicit.read_bytes(), b"payload")


class DictionaryGenerationUtilityTests(unittest.TestCase):
    def test_katakana_term_detection(self):
        self.assertTrue(_is_katakana_only("ゲーム"))
        self.assertTrue(_is_katakana_only("ティー・タイム"))
        self.assertFalse(_is_katakana_only("ア"))
        self.assertFalse(_is_katakana_only("ゲームA"))

    def test_regex_term_extraction(self):
        terms = _extract_regex_terms("今日はティータイム。スーパーへ行く。")
        self.assertIn("ティータイム", terms)
        self.assertIn("スーパー", terms)

    def test_sentence_selection_covers_terms_with_limit(self):
        sentences = [{"勇者", "城"}, {"魔王", "城"}, {"勇者", "剣"}, {"村"}]
        selected = solve_sentence_selection(sentences, max_select=2, name_set={"魔王"})
        self.assertLessEqual(len(selected), 2)
        self.assertIn(1, selected)
        self.assertEqual(len(selected), len(set(selected)))

    def test_sentence_selection_handles_empty_input(self):
        self.assertEqual(solve_sentence_selection([], max_select=3), [])


if __name__ == "__main__":
    unittest.main()
