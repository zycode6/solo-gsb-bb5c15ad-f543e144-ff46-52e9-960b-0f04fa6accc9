"""Unit tests for the pure pre-merge impact analysis (no dependencies)."""

import unittest

from app.analysis import (
    CATEGORY_CONFLICT,
    CATEGORY_LEFT_ONLY,
    CATEGORY_RIGHT_ONLY,
    CATEGORY_SAME_CHANGE,
    analyze_merge,
)


def by_pointer(result):
    return {r["pointer"]: r for r in result["records"]}


def cats(result):
    return {r["pointer"]: r["category"] for r in result["records"]}


class AnalysisClassificationTest(unittest.TestCase):
    def test_readme_style_triple(self):
        result = analyze_merge(
            {"replicas": 1, "db": {"host": "a", "port": 5432}},
            {"replicas": 3, "db": {"host": "b", "port": 5432}},
            {"replicas": 1, "db": {"host": "c", "port": 5433}},
        )
        self.assertEqual(
            cats(result),
            {
                "/replicas": CATEGORY_LEFT_ONLY,
                "/db/port": CATEGORY_RIGHT_ONLY,
                "/db/host": CATEGORY_CONFLICT,
            },
        )
        self.assertEqual(
            result["counts"],
            {"left_only": 1, "right_only": 1, "same_change": 0, "conflict": 1},
        )
        self.assertEqual(result["total"], 3)
        self.assertTrue(result["would_conflict"])
        self.assertEqual(result["conflict_count"], 1)

    def test_records_are_sorted_by_pointer(self):
        result = analyze_merge(
            {"z": 1, "a": {"b": 1, "a": 1, "c": 1}},
            {"z": 2, "a": {"b": 2, "a": 2, "c": 1}},
            {"z": 3, "a": {"b": 3, "a": 3, "c": 2}},
        )
        pointers = [r["pointer"] for r in result["records"]]
        self.assertEqual(pointers, sorted(pointers))
        self.assertEqual(
            pointers, ["/a/a", "/a/b", "/a/c", "/z"],
        )

    def test_left_and_right_only(self):
        result = analyze_merge(
            {"a": 1, "b": 2, "c": 3},
            {"a": 10, "b": 2, "c": 3},
            {"a": 1, "b": 20, "c": 3},
        )
        self.assertEqual(cats(result), {"/a": CATEGORY_LEFT_ONLY, "/b": CATEGORY_RIGHT_ONLY})
        record = by_pointer(result)["/a"]
        self.assertEqual((record["base"], record["left"], record["right"]), (1, 10, 1))
        self.assertEqual(record["missing"], [])

    def test_same_change_both_sides_identical(self):
        result = analyze_merge({"a": 1}, {"a": 5}, {"a": 5})
        self.assertEqual(cats(result), {"/a": CATEGORY_SAME_CHANGE})
        self.assertFalse(result["would_conflict"])

    def test_same_change_whole_documents_identical_descends_per_pointer(self):
        # Both sides made the same object edit: the in-place changes surface
        # per JSON Pointer rather than as one coarse root record.
        result = analyze_merge(
            {"a": 1, "b": 2}, {"a": 9, "b": 9}, {"a": 9, "b": 9}
        )
        self.assertEqual(
            cats(result),
            {"/a": CATEGORY_SAME_CHANGE, "/b": CATEGORY_SAME_CHANGE},
        )
        self.assertFalse(result["would_conflict"])

    def test_both_deleted_is_same_change(self):
        result = analyze_merge({"a": 1, "b": 2}, {"b": 2}, {"b": 2})
        record = by_pointer(result)["/a"]
        self.assertEqual(record["category"], CATEGORY_SAME_CHANGE)
        self.assertEqual(record["missing"], ["left", "right"])
        # Missing sides serialise as null but are marked in ``missing``.
        self.assertIsNone(record["left"])
        self.assertIsNone(record["right"])

    def test_one_side_deleted_other_unchanged_is_side_only(self):
        result = analyze_merge({"a": 1, "b": 2}, {"b": 2}, {"a": 1, "b": 2})
        record = by_pointer(result)["/a"]
        self.assertEqual(record["category"], CATEGORY_LEFT_ONLY)
        self.assertEqual(record["missing"], ["left"])

    def test_delete_vs_modify_conflicts(self):
        result = analyze_merge({"a": 1}, {}, {"a": 9})
        record = by_pointer(result)["/a"]
        self.assertEqual(record["category"], CATEGORY_CONFLICT)
        self.assertEqual(record["missing"], ["left"])
        self.assertEqual(record["right"], 9)

    def test_missing_distinct_from_explicit_null(self):
        # Left sets explicit null, right deletes: not the same change.
        result = analyze_merge({"k": 1}, {"k": None}, {})
        record = by_pointer(result)["/k"]
        self.assertEqual(record["category"], CATEGORY_CONFLICT)
        self.assertIsNone(record["left"])
        self.assertIsNone(record["right"])
        self.assertEqual(record["missing"], ["right"])

    def test_both_explicit_null_is_same_change(self):
        result = analyze_merge({"k": 1}, {"k": None}, {"k": None})
        record = by_pointer(result)["/k"]
        self.assertEqual(record["category"], CATEGORY_SAME_CHANGE)
        self.assertEqual(record["missing"], [])

    def test_both_added_same_key_same_value(self):
        result = analyze_merge({}, {"x": 1}, {"x": 1})
        record = by_pointer(result)["/x"]
        self.assertEqual(record["category"], CATEGORY_SAME_CHANGE)
        self.assertEqual(record["missing"], ["base"])

    def test_parent_deletion_reports_only_minimal_root(self):
        # Left deletes the whole subtree; right modifies one leaf: one
        # conflict record at /db, no per-child records.
        result = analyze_merge(
            {"db": {"host": "a", "port": 5432}, "keep": 1},
            {"keep": 1},
            {"db": {"host": "c", "port": 5433}, "keep": 1},
        )
        self.assertEqual(list(by_pointer(result)), ["/db"])
        record = by_pointer(result)["/db"]
        self.assertEqual(record["category"], CATEGORY_CONFLICT)
        self.assertEqual(record["missing"], ["left"])

    def test_single_side_subtree_delete_is_one_record(self):
        result = analyze_merge(
            {"db": {"host": "a", "port": 5432}},
            {},
            {"db": {"host": "a", "port": 5432}},
        )
        self.assertEqual(list(by_pointer(result)), ["/db"])
        self.assertEqual(cats(result), {"/db": CATEGORY_LEFT_ONLY})

    def test_type_replacement_reports_only_minimal_root(self):
        # Left changes a value inside the object, right replaces the whole
        # object with a scalar: conflict at /x, children never enumerated.
        result = analyze_merge({"x": {"y": 1}}, {"x": {"y": 2}}, {"x": "str"})
        self.assertEqual(list(by_pointer(result)), ["/x"])
        self.assertEqual(cats(result), {"/x": CATEGORY_CONFLICT})

    def test_array_change_is_opaque_minimal_root(self):
        result = analyze_merge(
            {"l": [1, 2]}, {"l": [1, 2, 3]}, {"l": [1, 2, 4]}
        )
        self.assertEqual(list(by_pointer(result)), ["/l"])
        self.assertEqual(cats(result), {"/l": CATEGORY_CONFLICT})

    def test_single_side_array_change_one_record(self):
        result = analyze_merge({"l": [1]}, {"l": [1, 2]}, {"l": [1]})
        self.assertEqual(cats(result), {"/l": CATEGORY_LEFT_ONLY})

    def test_root_level_conflict(self):
        result = analyze_merge(1, 2, 3)
        self.assertEqual(
            [r["pointer"] for r in result["records"]], [""]
        )
        self.assertEqual(cats(result), {"": CATEGORY_CONFLICT})
        self.assertTrue(result["would_conflict"])

    def test_root_level_single_side_change(self):
        result = analyze_merge(1, 2, 1)
        self.assertEqual(cats(result), {"": CATEGORY_LEFT_ONLY})

    def test_no_changes_empty_analysis(self):
        doc = {"a": {"nested": [1, 2]}, "b": 2}
        result = analyze_merge(doc, doc, doc)
        self.assertEqual(result["records"], [])
        self.assertEqual(result["total"], 0)
        self.assertFalse(result["would_conflict"])
        self.assertEqual(result["conflict_count"], 0)
        self.assertEqual(
            result["counts"],
            {"left_only": 0, "right_only": 0, "same_change": 0, "conflict": 0},
        )

    def test_pointer_escaping_in_records(self):
        result = analyze_merge(
            {"a/b": 1, "c~d": 1},
            {"a/b": 2, "c~d": 1},
            {"a/b": 1, "c~d": 2},
        )
        self.assertIn("/a~1b", cats(result))
        self.assertIn("/c~0d", cats(result))

    def test_objects_that_diverge_but_share_leaf_edits_recurse(self):
        # Different leaf changes merge recursively; the shared edit to /o/a
        # is a same_change and the right-only /o/b is adopted.
        result = analyze_merge(
            {"o": {"a": 0, "b": 0}},
            {"o": {"a": 5, "b": 0}},
            {"o": {"a": 5, "b": 7}},
        )
        self.assertEqual(
            cats(result), {"/o/a": CATEGORY_SAME_CHANGE, "/o/b": CATEGORY_RIGHT_ONLY}
        )
        self.assertFalse(result["would_conflict"])

    def test_base_missing_marker_on_added_keys(self):
        result = analyze_merge(
            {"keep": 1},
            {"keep": 1, "new": {"x": 1}},
            {"keep": 1, "new": {"x": 2}},
        )
        record = by_pointer(result)["/new/x"]
        self.assertEqual(record["category"], CATEGORY_CONFLICT)
        self.assertEqual(record["missing"], ["base"])

    def test_would_conflict_false_when_only_clean_changes(self):
        result = analyze_merge(
            {"a": 1, "b": 1},
            {"a": 2, "b": 1},
            {"a": 2, "b": 3},
        )
        self.assertFalse(result["would_conflict"])
        self.assertEqual(result["conflict_count"], 0)
        self.assertEqual(result["counts"]["conflict"], 0)

    def test_repeated_calls_are_identical(self):
        import json

        args = (
            {"z": 1, "a": {"y": 1, "x": [1, {"k": "v"}]}},
            {"z": 2, "a": {"y": 1, "x": [2, {"k": "w"}]}},
            {"z": 3, "a": {"y": 9, "x": [1, {"k": "v"}]}},
        )
        first = json.dumps(analyze_merge(*args), sort_keys=True)
        for _ in range(5):
            self.assertEqual(
                first, json.dumps(analyze_merge(*args), sort_keys=True)
            )

    def test_conflict_records_match_merge_engine(self):
        # The conflict classification must agree with merge3 on a mixed case.
        from app.merge import merge3

        base = {"a": 1, "b": {"x": 1}, "c": [1], "d": 1}
        left = {"a": 2, "b": {"x": 2}, "c": [2], "d": 1}
        right = {"a": 3, "b": {"x": 3}, "c": [1], "d": 9}
        result = analyze_merge(base, left, right)
        _, conflicts = merge3(base, left, right)
        engine_pointers = sorted(c["pointer"] for c in conflicts)
        analysis_pointers = sorted(
            r["pointer"] for r in result["records"]
            if r["category"] == CATEGORY_CONFLICT
        )
        self.assertEqual(engine_pointers, analysis_pointers)
        self.assertTrue(result["would_conflict"])


if __name__ == "__main__":
    unittest.main()
