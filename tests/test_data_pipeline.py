import csv
import json
import tempfile
import unittest
from pathlib import Path

import orjson

from GalTransl.CSentense import CSentense
from GalTransl.CSerialize import (
    save_json,
    save_transList_to_json_cn,
    update_json_with_transList,
)
from GalTransl.CSplitter import (
    DictionaryCombiner,
    DictionaryCountSplitter,
    EqualPartsSplitter,
    SplitChunkMetadata,
)
from GalTransl.Dictionary import CNormalDic, CGptDict, ifWord
from GalTransl.Loader import load_transList
from GalTransl.Name import (
    _load_existing_dst_names,
    extract_names_from_dir,
    extract_names_from_project,
    write_name_table_csv,
)
from plugins.file_galtransl_json.file_galtransl_json import file_plugin
from plugins.text_common_normalfix.text_common_normalfix import text_common_normalfix
from prompt2srt import (
    format_result,
    format_result_lrc,
    make_lrc,
    make_srt,
    merge_lrc_files,
)
from srt2prompt import make_prompt, merge_srt_files


class SentenceAndLoaderTests(unittest.TestCase):
    def test_sentence_state_and_speaker_names(self):
        sentence = CSentense("原文", ["甲", "乙"], 7)
        self.assertEqual(sentence.pre_jp, "原文")
        self.assertEqual(sentence.get_speaker_name(), "甲/乙")
        self.assertTrue(sentence.is_dialogue)
        with self.assertRaises(AttributeError):
            sentence.pre_jp = "修改"

    def test_dialogue_symbols_are_hidden_and_recovered(self):
        sentence = CSentense("『「台词」』")
        sentence.analyse_dialogue(dia_format="角色：#句子")
        self.assertEqual(sentence.post_jp, "角色：台词")
        self.assertEqual(sentence.left_symbol, "『「")
        self.assertEqual(sentence.right_symbol, "」』")
        sentence.post_zh = "译文"
        sentence.recover_dialogue_symbol()
        self.assertEqual(sentence.post_zh, "『「译文」』")

    def test_split_dialogue_links_two_sentences(self):
        sentences, _ = load_transList([
            {"name": "角色", "message": "「前半"},
            {"message": "后半」"},
        ])
        sentences[0].analyse_dialogue()
        self.assertEqual(sentences[0].post_jp, "前半")
        self.assertEqual(sentences[1].post_jp, "后半")
        self.assertTrue(sentences[1].is_dialogue)
        self.assertEqual(sentences[1].speaker, "角色")

    def test_loads_list_json_text_and_file_and_links_context(self):
        rows = [{"name": "A", "message": "一"}, {"names": ["B"], "message": "二", "index": 9}]
        for source_factory in (
            lambda _: rows,
            lambda _: json.dumps(rows, ensure_ascii=False),
            lambda root: str(root / "input.json"),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                (root / "input.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
                result, original = load_transList(source_factory(root))
                self.assertEqual([item.index for item in result], [1, 9])
                self.assertIs(result[0].next_tran, result[1])
                self.assertIs(result[1].prev_tran, result[0])
                self.assertEqual(original, rows)

    def test_loader_rejects_invalid_inputs(self):
        with self.assertRaises(TypeError):
            load_transList(123)
        with self.assertRaises(ValueError):
            load_transList("{}")
        with self.assertRaises(ValueError):
            load_transList(["not a mapping"])
        with self.assertRaises(ValueError):
            load_transList([{"name": "missing message"}])


class SerializationTests(unittest.TestCase):
    def test_serializes_translations_and_replaces_names(self):
        trans, _ = load_transList([
            {"name": "A", "message": "一"},
            {"names": ["B", "C"], "message": "二"},
            {"message": "三"},
        ])
        for index, item in enumerate(trans, 1):
            item.post_zh = f"译{index}"
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "output.json"
            save_transList_to_json_cn(trans, str(output), {"A": "甲", "B": "乙"})
            data = orjson.loads(output.read_bytes())
        self.assertEqual(data[0], {"name": "甲", "message": "译1"})
        self.assertEqual(data[1], {"names": ["乙", "C"], "message": "译2"})
        self.assertEqual(data[2], {"message": "译3"})

    def test_updates_only_rows_matching_original_source(self):
        trans, original = load_transList([
            {"name": "A", "message": "一", "extra": 1},
            {"message": "二"},
        ])
        trans[0].post_zh = "壹"
        trans[1].post_zh = "贰"
        original[1]["message"] = "changed"
        result = update_json_with_transList(trans, original, {"A": "甲"})
        self.assertEqual(result[0]["message"], "壹")
        self.assertEqual(result[0]["name"], "甲")
        self.assertEqual(result[0]["extra"], 1)
        self.assertEqual(result[1]["message"], "changed")

    def test_save_json_writes_unicode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "data.json"
            save_json(str(output), [{"message": "中文"}])
            self.assertEqual(orjson.loads(output.read_bytes()), [{"message": "中文"}])


class SplitterTests(unittest.TestCase):
    @staticmethod
    def rows(count):
        return [{"index": str(i + 10), "message": f"line-{i}"} for i in range(count)]

    def setUp(self):
        SplitChunkMetadata.clear_file_finished_chunk()

    def test_dictionary_count_splitter_adds_overlap_and_metadata(self):
        chunks = DictionaryCountSplitter(2, cross_num=1).split(self.rows(5), "story.json")
        self.assertEqual(len(chunks), 3)
        self.assertEqual([(c.start_index, c.end_index) for c in chunks], [(0, 2), (2, 4), (4, 5)])
        self.assertEqual([c.chunk_size for c in chunks], [3, 4, 2])
        self.assertTrue(all(c.total_chunks == 3 for c in chunks))
        self.assertEqual(chunks[1].trans_list[0].runtime_index, 11)

    def test_equal_parts_splitter_distributes_remainder(self):
        chunks = EqualPartsSplitter(3).split(self.rows(8), "story.json")
        self.assertEqual([c.chunk_non_cross_size for c in chunks], [3, 3, 2])
        self.assertEqual([c.start_index for c in chunks], [0, 3, 6])

    def test_combiner_removes_overlapping_rows(self):
        rows = self.rows(5)
        chunks = DictionaryCountSplitter(2, cross_num=1).split(rows, "story.json")
        combined_trans, combined_json = DictionaryCombiner.combine(list(reversed(chunks)))
        self.assertEqual([row["message"] for row in combined_json], [row["message"] for row in rows])
        self.assertEqual([item.pre_jp for item in combined_trans], [row["message"] for row in rows])

    def test_finished_chunk_tracking_is_per_file(self):
        chunks = DictionaryCountSplitter(1).split(self.rows(2), "a.json")
        chunks[0].update_file_finished_chunk()
        self.assertFalse(chunks[0].is_file_finished())
        chunks[1].update_file_finished_chunk()
        self.assertTrue(chunks[1].is_file_finished())
        self.assertEqual(len(chunks[0].get_file_finished_chunks()), 2)


class DictionaryTests(unittest.TestCase):
    def test_if_word_flags(self):
        word = ifWord(">!prefix<")
        self.assertTrue(word.startswith_flag)
        self.assertTrue(word.endswith_flag)
        self.assertTrue(word.without_flag)
        self.assertEqual(word.word, "prefix")

    def test_normal_dictionary_replacement_modes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "normal.txt"
            path.write_text("长词\tL\n词\tW\n^^开头\t首\n1^重复\t一次\nmono\t独白\t旁白\n", encoding="utf-8")
            dictionary = CNormalDic([str(path)])
            dictionary.sort_dic()
            tran = CSentense("", "")
            result = dictionary.do_replace("开头 长词 词 重复重复 独白", tran)
            self.assertEqual(result, "首 L W 一次重复 旁白")
            tran.is_dialogue = True
            self.assertEqual(dictionary.do_replace("独白", tran), "独白")

    def test_conditional_dictionary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "conditional.txt"
            path.write_text("pre_src\t勇者[and]!魔王\t剣\t剑\n", encoding="utf-8")
            dictionary = CNormalDic([str(path)])
            matching = CSentense("勇者出发")
            blocked = CSentense("勇者与魔王")
            self.assertEqual(dictionary.do_replace("剣", matching), "剑")
            self.assertEqual(dictionary.do_replace("剣", blocked), "剣")

    def test_gpt_dictionary_load_prompt_and_lookup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gpt.txt"
            path.write_text("勇者\tHero\t人名\n魔王->Demon King#称号\n", encoding="utf-8")
            dictionary = CGptDict([str(path)])
            trans = [CSentense("勇者は魔王を倒す")]
            prompt = dictionary.gen_prompt(trans, type="gpt")
            self.assertIn("# Glossary", prompt)
            self.assertIn("| 勇者 | Hero | 人名 |", prompt)
            self.assertIn("| 魔王 | Demon King | 称号 |", prompt)
            self.assertEqual(dictionary.get_dst("勇者"), "Hero")


class SubtitleConversionTests(unittest.TestCase):
    def test_timestamp_formatting(self):
        self.assertEqual(format_result(3661.234), "01:01:01,234")
        self.assertEqual(format_result_lrc(61.234), "01:01.234")

    def test_json_srt_prompt_round_trip(self):
        rows = [{"start": 1.25, "end": 2.5, "message": "第一行\n第二行"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input.json"
            srt = root / "output.srt"
            prompt = root / "roundtrip.json"
            source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            make_srt(str(source), str(srt))
            parsed = make_prompt(str(srt))
            make_prompt(str(srt), str(prompt))
            self.assertEqual(parsed, rows)
            self.assertEqual(json.loads(prompt.read_text(encoding="utf-8")), rows)

    def test_lrc_creation_and_offset_merge(self):
        rows = [{"start": 1.0, "end": 2.0, "message": "A"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input.json"
            first = root / "first.lrc"
            second = root / "second.lrc"
            merged = root / "merged.lrc"
            source.write_text(json.dumps(rows), encoding="utf-8")
            make_lrc(str(source), str(first))
            second.write_text("[00:02.000] B\n", encoding="utf-8")
            merge_lrc_files([str(first), str(second)], str(merged), duration=10)
            lines = merged.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines, ["[00:01.000] A", "[00:12.000] B"])

    def test_srt_merge_offsets_and_renumbers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a.srt"
            second = root / "b.srt"
            merged = root / "merged.srt"
            first.write_text("1\n00:00:01,000 --> 00:00:02,000\nA\n", encoding="utf-8")
            second.write_text("8\n00:00:01,000 --> 00:00:02,000\nB\n", encoding="utf-8")
            merge_srt_files([str(first), str(second)], str(merged), duration=10)
            parsed = make_prompt(str(merged))
            self.assertEqual([row["start"] for row in parsed], [1.0, 11.0])
            self.assertIn("\n2\n", merged.read_text(encoding="utf-8-sig"))


class PluginAndNameTests(unittest.TestCase):
    def test_json_file_plugin_round_trip_and_source_copy(self):
        plugin = file_plugin()
        plugin.gtp_init(
            {"Core": {"Name": "JSON"}, "Settings": {"output_with_src": True}},
            {},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input.json"
            output = root / "output.json"
            source.write_text('[{"message":"原文"}]', encoding="utf-8")
            rows = plugin.load_file(str(source))
            self.assertEqual(rows[0]["src_msg"], "原文")
            rows[0]["message"] = "译文"
            plugin.save_file(str(output), rows)
            self.assertEqual(orjson.loads(output.read_bytes())[0]["message"], "译文")
            with self.assertRaises(TypeError):
                plugin.load_file(str(root / "input.txt"))

    def test_text_normalization_plugin_preserves_outer_spacing(self):
        plugin = text_common_normalfix()
        tran = CSentense("　 原文 \\n")
        tran = plugin.before_src_processed(tran)
        self.assertEqual(tran.post_jp, "原文")
        self.assertEqual(tran.left_symbol, "　 ")
        self.assertEqual(tran.right_symbol, " \\n")
        tran.post_zh = "，译文。"
        tran = plugin.before_dst_processed(tran)
        self.assertEqual(tran.post_zh, "译文")

    def test_name_extraction_csv_write_and_existing_translation_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "gt_input"
            input_dir.mkdir()
            (input_dir / "a.json").write_text(
                json.dumps([{"name": "甲", "message": "1"}, {"names": ["甲", "乙"], "message": "2"}], ensure_ascii=False),
                encoding="utf-8",
            )
            (input_dir / "bad.json").write_text("not json", encoding="utf-8")
            expected = {"甲": 2, "乙": 1}
            self.assertEqual(extract_names_from_dir(str(input_dir)), expected)
            self.assertEqual(extract_names_from_project(str(root)), expected)

            table = root / "name替换表.csv"
            write_name_table_csv(str(table), expected, {"甲": "A"})
            self.assertEqual(_load_existing_dst_names(str(root)), {"甲": "A"})
            with table.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows[0], ["SRC_Name", "DST_Name", "Count"])


if __name__ == "__main__":
    unittest.main()
