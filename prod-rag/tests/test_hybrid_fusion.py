"""Tests for hybrid retrieval fusion.

The retriever merges a semantic list (Chroma, cosine similarity) with a keyword
list (Elasticsearch, BM25). Those two scores live on different scales -- cosine
is bounded to roughly [0, 1], BM25 is unbounded and routinely exceeds 20 -- so
any ranking that compares them directly is comparing incommensurable numbers.

These tests pin the properties fusion must have regardless of implementation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retriever import HybridRetriever  # noqa: E402


@pytest.fixture
def fuse():
    retriever = HybridRetriever()
    return retriever._merge_and_dedupe


def doc(doc_id, score, text=None):
    return {"id": doc_id, "score": score, "text": text or f"body of {doc_id}"}


def ids(results):
    return [r["id"] for r in results]


class TestFusionPreservesDocuments:
    def test_union_of_both_lists_is_returned(self, fuse):
        out = fuse([doc("a", 0.9), doc("b", 0.8)], [doc("c", 12.0), doc("d", 9.0)])
        assert sorted(ids(out)) == ["a", "b", "c", "d"]

    def test_documents_in_both_lists_appear_once(self, fuse):
        out = fuse([doc("a", 0.9), doc("shared", 0.5)], [doc("shared", 30.0), doc("b", 8.0)])
        assert ids(out).count("shared") == 1
        assert sorted(ids(out)) == ["a", "b", "shared"]

    def test_both_retrieval_sources_are_recorded(self, fuse):
        out = fuse([doc("shared", 0.5)], [doc("shared", 30.0)])
        assert sorted(out[0]["retrieval_sources"]) == ["chroma", "elasticsearch"]

    def test_empty_inputs_are_handled(self, fuse):
        assert fuse([], []) == []
        assert ids(fuse([doc("a", 0.9)], [])) == ["a"]
        assert ids(fuse([], [doc("z", 10.0)])) == ["z"]


class TestAgreementIsRewarded:
    def test_a_document_both_retrievers_found_outranks_single_source_hits(self, fuse):
        out = fuse(
            [doc("both", 0.6), doc("sem_only", 0.9)],
            [doc("both", 20.0), doc("kw_only", 25.0)],
        )
        assert ids(out)[0] == "both"


class TestKeywordOnlyResultsAreNotStarved:
    """The original ranking sorted by (source_count, semantic_score,
    keyword_score) as a tuple. Because semantic_score is compared before
    keyword_score and every Elasticsearch-only document has semantic_score 0.0,
    a near-irrelevant semantic hit outranked a perfect keyword match. That makes
    the "hybrid" retriever semantic-only in effect."""

    def test_keyword_only_hit_is_not_starved_behind_weak_semantic_hits(self, fuse):
        """Rank fusion ties the top of each list rather than ordering by raw
        score, so the assertion is that the keyword hit reaches the top group --
        not that it beats the top semantic hit, which would be reading more into
        incomparable scores than they support."""
        semantic = [doc(f"s{i}", 0.9 - i * 0.2) for i in range(4)]
        out = fuse(semantic, [doc("k_top", 45.0)])
        assert ids(out).index("k_top") <= 1

    def test_top_keyword_hit_outranks_all_but_the_top_semantic_hit(self, fuse):
        """Previously this landed dead last, behind all eight semantic hits."""
        semantic = [doc(f"s{i}", 0.9 - i * 0.1) for i in range(8)]
        out = fuse(semantic, [doc("k_top", 60.0)])
        assert ids(out).index("k_top") <= 1

    def test_keyword_rank_order_is_respected(self, fuse):
        out = fuse([], [doc("first", 50.0), doc("second", 30.0), doc("third", 10.0)])
        assert ids(out) == ["first", "second", "third"]

    def test_semantic_rank_order_is_respected(self, fuse):
        out = fuse([doc("first", 0.9), doc("second", 0.7), doc("third", 0.3)], [])
        assert ids(out) == ["first", "second", "third"]


class TestRankingIsScaleInvariant:
    """BM25 scores vary hugely with corpus and query length. Multiplying every
    keyword score by a constant must not reorder anything."""

    def test_scaling_keyword_scores_does_not_change_order(self, fuse):
        semantic = [doc("a", 0.8), doc("b", 0.4)]
        keyword = [doc("c", 10.0), doc("a", 5.0)]
        baseline = ids(fuse(semantic, keyword))
        scaled = ids(fuse(semantic, [doc("c", 1000.0), doc("a", 500.0)]))
        assert baseline == scaled
