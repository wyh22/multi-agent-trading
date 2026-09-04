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
