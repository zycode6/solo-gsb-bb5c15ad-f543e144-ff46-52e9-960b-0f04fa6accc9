"""Pre-merge impact analysis: classify every change without persisting.

Given the same ``base`` / ``left`` / ``right`` triple as :func:`app.merge.merge3`
(but without creating a merge order), this walks the three documents by JSON
Pointer and reports one change record per *minimal* affected root path:

  * ``left_only``    — only the left side diverged from base; auto-adopted
  * ``right_only``   — only the right side diverged from base; auto-adopted
  * ``same_change``  — both sides diverged identically (incl. both deleted);
                       auto-merged
  * ``conflict``     — the sides diverged differently; this is exactly where
                       the merge engine would leave a conflict placeholder

The walk mirrors the merge rules per path:

  * neither side changed relative to base -> no record and no descent;
  * exactly one side changed -> recurse while base and the changed value are
    both objects (individual in-place edits are reported at their own leaf
    pointers), but collapse to one record at the current path when the key is
    deleted, an array is replaced or the value changes type — the subtree is
    never enumerated, so its children are not double counted;
  * both sides changed identically -> same rule, reporting ``same_change``;
  * both sides changed differently -> objects recurse key-by-key (a child
    changed on only one side is reported as a side-only record); anything
    else (scalars, arrays, missing values, type changes) is one ``conflict``
    record at the minimal root.

Absent keys (``MISSING``) stay distinct from explicit JSON ``null`` via the
per-record ``missing`` list, mirroring the conflict records produced by the
merge engine. Everything here is pure and in-memory; repeated calls with the
same inputs return identical responses (records are sorted by JSON Pointer).
"""

from __future__ import annotations

from typing import Any

from .merge import MISSING, _eq, escape, merge3

CATEGORY_LEFT_ONLY = "left_only"
CATEGORY_RIGHT_ONLY = "right_only"
CATEGORY_SAME_CHANGE = "same_change"
CATEGORY_CONFLICT = "conflict"


def analyze_merge(base: Any, left: Any, right: Any) -> dict:
    """Return the stable, sorted change-record analysis of a merge triple.

    The response carries one record per minimal affected root path, the
    per-category counts and ``would_conflict``. It writes nothing; the
    conflict verdict is derived from the real merge engine so the dry run
    cannot drift from what ``POST /merges`` would produce.
    """
    records: list[dict] = []
    _analyze(base, left, right, "", records)
    records.sort(key=lambda r: r["pointer"])

    counts = {
        CATEGORY_LEFT_ONLY: 0,
        CATEGORY_RIGHT_ONLY: 0,
        CATEGORY_SAME_CHANGE: 0,
        CATEGORY_CONFLICT: 0,
    }
    for record in records:
        counts[record["category"]] += 1

    # Authoritative conflict verdict straight from the merge engine rather
    # than re-deriving it: the dry run must never disagree with a real merge.
    # The two conflict sets are structurally identical (same branches, same
    # recursion condition), so conflict_count also equals counts["conflict"].
    _, conflicts = merge3(base, left, right)
    return {
        "records": records,
        "counts": counts,
        "total": len(records),
        "would_conflict": bool(conflicts),
        "conflict_count": len(conflicts),
    }


def _record(
    category: str, path: str, base: Any, left: Any, right: Any
) -> dict:
    """Build one change record, preserving missing-vs-explicit-null."""
    return {
        "pointer": path,
        "category": category,
        "base": None if base is MISSING else base,
        "left": None if left is MISSING else left,
        "right": None if right is MISSING else right,
        "missing": [
            name
            for name, value in (("base", base), ("left", left), ("right", right))
            if value is MISSING
        ],
    }


def _descend(base: Any, left: Any, right: Any, path: str, records: list[dict]) -> None:
    """Continue the analysis one object level down, key by key."""
    base_obj = base if isinstance(base, dict) else {}
    left_obj = left if isinstance(left, dict) else {}
    right_obj = right if isinstance(right, dict) else {}
    # Sorted key union keeps traversal order deterministic regardless of
    # hash-seeded set iteration or the incoming key order.
    for key in sorted(base_obj.keys() | left_obj.keys() | right_obj.keys()):
        _analyze(
            base_obj.get(key, MISSING),
            left_obj.get(key, MISSING),
            right_obj.get(key, MISSING),
            path + "/" + escape(key),
            records,
        )


def _analyze(base: Any, left: Any, right: Any, path: str, records: list[dict]) -> None:
    left_changed = not _eq(left, base)
    right_changed = not _eq(right, base)

    # Neither side diverged: nothing to report, and identical untouched
    # subtrees are not enumerated path by path.
    if not left_changed and not right_changed:
        return

    if left_changed and right_changed:
        if _eq(left, right):
            # Both sides made the very same change. Recurse while the change
            # is an in-place object edit (shared nested edits are reported at
            # their own leaf pointers); deletions, arrays and type changes
            # collapse to one same_change record at the minimal root.
            if isinstance(base, dict) and isinstance(left, dict):
                _descend(base, left, right, path, records)
            else:
                records.append(_record(CATEGORY_SAME_CHANGE, path, base, left, right))
            return
        # The sides diverged differently. Objects recurse key-by-key exactly
        # like the merge engine (a child touched by only one side becomes a
        # side-only record down there); scalars, arrays, missing values and
        # type replacements are a single conflict at the minimal root, so the
        # replaced/deleted subtree is never double counted.
        if isinstance(left, dict) and isinstance(right, dict):
            _descend(base, left, right, path, records)
        else:
            records.append(_record(CATEGORY_CONFLICT, path, base, left, right))
        return

    # Exactly one side changed. Descend only while both the base and the new
    # value are objects (in-place edits surface at their own pointers); a
    # deletion, an array replacement or a type change is one side-only record
    # at this minimal root.
    changed = left if left_changed else right
    category = CATEGORY_LEFT_ONLY if left_changed else CATEGORY_RIGHT_ONLY
    if isinstance(base, dict) and isinstance(changed, dict):
        _descend(base, left, right, path, records)
    else:
        records.append(_record(category, path, base, left, right))
