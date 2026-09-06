from tradingagents.rag.evidence_pack import build_repair_queries, source_document_key
from tradingagents.rag.models import KnowledgeChunk
from tradingagents.rag.retriever import HybridKnowledgeRetriever, InMemoryKnowledgeStore

def _chunk(cid, date, text):
    return KnowledgeChunk(chunk_id=cid,doc_id=cid.split("::")[0],ticker="600000.SH",title=cid,text=text,publish_date=date,source="test",doc_type="announcement")

def test_rag_never_returns_future_document():
    chunks=[_chunk("old::0","2026-04-01","公司发布年度报告，经营现金流改善。"),_chunk("future::0","2026-09-01","未来公告：重大资产重组。")]
    retriever=HybridKnowledgeRetriever(InMemoryKnowledgeStore(chunks))
    hits=retriever.search("资产重组 现金流",ticker="600000.SH",as_of_date="2026-08-20",top_k=5)
    assert hits
    assert all(hit.chunk.publish_date <= "2026-08-20" for hit in hits)
    assert all(hit.chunk.chunk_id != "future::0" for hit in hits)

def test_hybrid_rag_prefers_lexically_relevant_evidence():
    chunks=[_chunk("a::0","2026-03-01","公司召开股东大会。"),_chunk("b::0","2026-03-02","经营现金流净额同比增长，现金流质量改善。")]
    retriever=HybridKnowledgeRetriever(InMemoryKnowledgeStore(chunks))
    hits=retriever.search("经营现金流",ticker="600000.SH",as_of_date="2026-08-20",top_k=1)
    assert hits[0].chunk.chunk_id == "b::0"


def test_repair_query_planner_focuses_known_evidence_gaps():
    queries = build_repair_queries(
        "缺少主营业务、装机容量、政策风险、弃风限电和补贴回款证据",
        max_queries=4,
    )
    assert len(queries) == 4
    assert any("装机容量" in query for query in queries)
    assert any("行业政策" in query for query in queries)
    assert any("弃风率" in query for query in queries)
    assert any("补贴应收" in query for query in queries)


def test_retrieval_limits_chunks_from_same_parent_document():
    chunks = [
        KnowledgeChunk(
            chunk_id="a-page1::0",
            doc_id="a-page1",
            ticker="600000.SH",
            title="年报第1页",
            text="风电装机容量 发电量 利用小时 运营规模",
            publish_date="2026-04-01",
            source="test",
            doc_type="annual_report",
            metadata={"file_hash": "annual-report-a"},
        ),
        KnowledgeChunk(
            chunk_id="a-page2::0",
            doc_id="a-page2",
            ticker="600000.SH",
            title="年报第2页",
            text="风电装机容量 发电量 利用小时 运营规模",
            publish_date="2026-04-01",
            source="test",
            doc_type="annual_report",
            metadata={"file_hash": "annual-report-a"},
        ),
        KnowledgeChunk(
            chunk_id="b::0",
            doc_id="b",
            ticker="600000.SH",
            title="经营公告",
            text="公司披露风电装机容量和年度发电量。",
            publish_date="2026-05-01",
            source="test",
            doc_type="announcement",
            metadata={"file_hash": "operations-b"},
        ),
    ]
    retriever = HybridKnowledgeRetriever(InMemoryKnowledgeStore(chunks))
    hits = retriever.search(
        "风电装机容量 发电量",
        ticker="600000.SH",
        as_of_date="2026-08-20",
        top_k=2,
        max_chunks_per_doc=1,
    )
    assert len(hits) == 2
    assert len({source_document_key(hit.chunk) for hit in hits}) == 2
