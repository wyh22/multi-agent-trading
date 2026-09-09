from tradingagents.evaluation.rag_retrieval import (
    EvidenceJudgment,
    RetrievalCase,
    evaluate_retrieval_hits,
)
from tradingagents.rag.models import KnowledgeChunk, RetrievalHit
from tradingagents.rag.retriever import (
    HybridKnowledgeRetriever,
    InMemoryKnowledgeStore,
)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc: str,
    page: int,
    publish_date: str = "2026-01-01",
):
    return KnowledgeChunk(
        chunk_id=chunk_id,
        doc_id=doc,
        ticker="600519.SH",
        title=f"doc {doc}",
        text=text,
        publish_date=publish_date,
        scope_type="company",
        scope_key="600519.SH",
        metadata={
            "file_hash": doc,
            "page": page,
            "publish_date_verified": True,
        },
    )


def test_retriever_supports_dense_bm25_and_hybrid_ablation():
    chunks = [
        _chunk("c1", "主营业务 白酒 茅台酒 系列酒", doc="d1", page=1),
        _chunk("c2", "资本开支 在建工程 产能 建设", doc="d2", page=2),
        _chunk("c3", "风险 原材料 市场竞争", doc="d3", page=3),
    ]
    retriever = HybridKnowledgeRetriever(InMemoryKnowledgeStore(chunks))

    dense = retriever.search(
        "资本开支 产能",
        ticker="600519.SH",
        as_of_date="2026-09-07",
        top_k=2,
        strategy="dense",
        use_reranker=False,
    )
    bm25 = retriever.search(
        "资本开支 产能",
        ticker="600519.SH",
        as_of_date="2026-09-07",
        top_k=2,
        strategy="bm25",
        use_reranker=False,
    )
    hybrid = retriever.search(
        "资本开支 产能",
        ticker="600519.SH",
        as_of_date="2026-09-07",
        top_k=2,
        strategy="hybrid",
        use_reranker=False,
    )

    assert dense and dense[0].dense_score is not None
    assert bm25 and bm25[0].bm25_score is not None
    assert hybrid
    assert any(
        hit.dense_score is not None or hit.bm25_score is not None
        for hit in hybrid
    )


def test_retrieval_metrics_compute_recall_mrr_ndcg_and_document_recall():
    case = RetrievalCase(
        case_id="case-1",
        ticker="600519.SH",
        query="资本开支",
        as_of_date="2026-09-07",
    )
    relevant_a = _chunk("c1", "资本开支", doc="annual", page=10)
    irrelevant = _chunk("c2", "其他信息", doc="other", page=3)
    relevant_b = _chunk("c3", "在建工程", doc="semi", page=20)
    hits = [
        RetrievalHit(chunk=irrelevant, score=0.9),
        RetrievalHit(chunk=relevant_a, score=0.8),
        RetrievalHit(chunk=relevant_b, score=0.7),
    ]
    judgments = [
        EvidenceJudgment(
            case_id="case-1",
            relevance=3,
            chunk_id="c1",
            document_key="annual",
            page=10,
        ),
        EvidenceJudgment(
            case_id="case-1",
            relevance=2,
            chunk_id="c3",
            document_key="semi",
            page=20,
        ),
    ]

    row = evaluate_retrieval_hits(
        case,
        hits,
        judgments,
        strategy="hybrid",
        ks=(1, 3),
    ).to_dict()

    assert row["recall@1"] == 0.0
    assert row["recall@3"] == 1.0
    assert row["mrr@3"] == 0.5
    assert row["document_recall@3"] == 1.0
    assert 0.0 < row["ndcg@3"] <= 1.0
    assert row["pit_violations"] == 0


def test_metric_reports_future_document_as_pit_violation():
    case = RetrievalCase(
        case_id="pit",
        ticker="600519.SH",
        query="风险",
        as_of_date="2026-06-30",
    )
    future = _chunk(
        "future",
        "风险",
        doc="future-doc",
        page=1,
        publish_date="2026-08-15",
    )
    row = evaluate_retrieval_hits(
        case,
        [RetrievalHit(chunk=future, score=1.0)],
        [
            EvidenceJudgment(
                case_id="pit",
                relevance=1,
                chunk_id="future",
                document_key="future-doc",
                page=1,
            )
        ],
        strategy="dense",
        ks=(1,),
    ).to_dict()

    assert row["pit_violations"] == 1
