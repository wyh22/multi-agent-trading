import pandas as pd

from tradingagents.rag.bootstrap import _normalize_url, select_high_value_disclosures
from tradingagents.rag.evidence_pack import (
    build_retrieval_queries,
    source_document_key,
)
from tradingagents.rag.models import KnowledgeChunk
from tradingagents.rag.retriever import (
    HybridKnowledgeRetriever,
    InMemoryKnowledgeStore,
)
from tradingagents.rag.scope import SHARED_TICKER


def _chunk(cid, publish_date, text):
    return KnowledgeChunk(
        chunk_id=cid,
        doc_id=cid.split("::")[0],
        ticker="600000.SH",
        title=cid,
        text=text,
        publish_date=publish_date,
        source="test",
        doc_type="announcement",
    )


def test_rag_never_returns_future_document():
    chunks = [
        _chunk(
            "old::0",
            "2026-04-01",
            "公司发布年度报告，经营现金流改善。",
        ),
        _chunk(
            "future::0",
            "2026-09-01",
            "未来公告：重大资产重组。",
        ),
    ]
    retriever = HybridKnowledgeRetriever(InMemoryKnowledgeStore(chunks))
    hits = retriever.search(
        "资产重组 现金流",
        ticker="600000.SH",
        as_of_date="2026-08-20",
        top_k=5,
    )
    assert hits
    assert all(hit.chunk.publish_date <= "2026-08-20" for hit in hits)
    assert all(hit.chunk.chunk_id != "future::0" for hit in hits)


def test_hybrid_rag_prefers_lexically_relevant_evidence():
    chunks = [
        _chunk("a::0", "2026-03-01", "公司召开股东大会。"),
        _chunk(
            "b::0",
            "2026-03-02",
            "经营现金流净额同比增长，现金流质量改善。",
        ),
    ]
    retriever = HybridKnowledgeRetriever(InMemoryKnowledgeStore(chunks))
    hits = retriever.search(
        "经营现金流",
        ticker="600000.SH",
        as_of_date="2026-08-20",
        top_k=1,
    )
    assert hits[0].chunk.chunk_id == "b::0"


def test_retrieval_query_planner_is_sector_agnostic():
    queries = build_retrieval_queries(
        "补查主营业务、客户集中度、估值、资本开支、行业政策和主要风险",
        max_queries=6,
    )
    joined = "\n".join(queries)
    assert "主营业务" in joined
    assert "竞争格局" in joined
    assert "估值" in joined
    assert "资本开支" in joined
    assert "政策监管" in joined
    assert "经营风险" in joined
    assert "风电" not in joined
    assert "白酒" not in joined


def test_retrieval_limits_chunks_from_same_parent_document():
    chunks = [
        KnowledgeChunk(
            chunk_id="a-page1::0",
            doc_id="a-page1",
            ticker="600000.SH",
            title="年报第1页",
            text="主营业务 产能 订单 客户结构",
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
            text="主营业务 产能 订单 客户结构",
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
            text="公司披露新增产能和重大订单。",
            publish_date="2026-05-01",
            source="test",
            doc_type="announcement",
            metadata={"file_hash": "operations-b"},
        ),
    ]
    retriever = HybridKnowledgeRetriever(InMemoryKnowledgeStore(chunks))
    hits = retriever.search(
        "主营业务 产能 订单",
        ticker="600000.SH",
        as_of_date="2026-08-20",
        top_k=2,
        max_chunks_per_doc=1,
    )
    assert len(hits) == 2
    assert len({source_document_key(hit.chunk) for hit in hits}) == 2


def test_company_search_includes_shared_regulation_scope():
    chunks = [
        KnowledgeChunk(
            chunk_id="company::0",
            doc_id="company",
            ticker="600000.SH",
            title="公司经营公告",
            text="公司披露主营业务经营情况。",
            publish_date="2026-05-01",
            source="company",
            doc_type="announcement",
        ),
        KnowledgeChunk(
            chunk_id="regulation::0",
            doc_id="regulation",
            ticker=SHARED_TICKER,
            title="上市公司行业监管规则",
            text="行业监管规则涉及信息披露与经营合规要求。",
            publish_date="2026-04-15",
            source="regulator",
            doc_type="regulation",
            scope_type="regulation",
            scope_key="CN",
        ),
    ]
    retriever = HybridKnowledgeRetriever(InMemoryKnowledgeStore(chunks))
    hits = retriever.search(
        "行业监管规则",
        ticker="600000.SH",
        as_of_date="2026-08-20",
        top_k=2,
    )
    assert any(hit.chunk.scope_type == "regulation" for hit in hits)


def test_company_search_automatically_adds_known_industry_scope():
    chunks = [
        KnowledgeChunk(
            chunk_id="company::0",
            doc_id="company",
            ticker="600000.SH",
            title="公司年报",
            text="公司主营消费品业务。",
            publish_date="2026-04-01",
            source="company",
            doc_type="annual_report",
            industry="消费品",
        ),
        KnowledgeChunk(
            chunk_id="industry::0",
            doc_id="industry",
            ticker=SHARED_TICKER,
            title="消费品行业竞争格局",
            text="行业竞争格局、市场份额和渠道变化。",
            publish_date="2026-03-20",
            source="industry-source",
            doc_type="industry_report",
            scope_type="industry",
            scope_key="消费品",
            industry="消费品",
        ),
    ]
    retriever = HybridKnowledgeRetriever(InMemoryKnowledgeStore(chunks))
    hits = retriever.search(
        "行业竞争格局 市场份额",
        ticker="600000.SH",
        as_of_date="2026-08-20",
        top_k=2,
    )
    assert any(hit.chunk.scope_type == "industry" for hit in hits)


def test_autumn_bootstrap_selects_high_value_generic_filings():
    disclosures = pd.DataFrame(
        [
            {
                "公告时间": "2026-08-29",
                "公告标题": "2026年半年度报告",
                "公告链接": "https://static.cninfo.com.cn/semi.pdf",
            },
            {
                "公告时间": "2026-07-15",
                "公告标题": "投资者关系活动记录表",
                "公告链接": "https://static.cninfo.com.cn/ir.pdf",
            },
            {
                "公告时间": "2026-04-20",
                "公告标题": "2025年年度报告摘要",
                "公告链接": "https://static.cninfo.com.cn/summary.pdf",
            },
            {
                "公告时间": "2026-04-20",
                "公告标题": "2025年年度报告",
                "公告链接": "https://static.cninfo.com.cn/annual.pdf",
            },
        ]
    )
    selected = select_high_value_disclosures(
        disclosures,
        ticker="600000.SH",
        annual_year=2025,
        interim_year=2026,
        max_docs=3,
    )
    assert [item.doc_type for item in selected] == [
        "annual_report",
        "semiannual_report",
        "investor_relation",
    ]
    assert all("summary.pdf" not in item.url for item in selected)


def test_autumn_bootstrap_converts_cninfo_detail_url_to_static_pdf():
    detail_url = (
        "http://www.cninfo.com.cn/new/disclosure/detail?"
        "stockCode=600519&announcementId=1225114741&"
        "orgId=gssh0600519&announcementTime=2026-04-17%2000:00:00"
    )
    assert _normalize_url(detail_url) == (
        "https://static.cninfo.com.cn/finalpage/2026-04-17/1225114741.PDF"
    )


def test_autumn_bootstrap_falls_back_to_q1_without_ir_record():
    disclosures = pd.DataFrame(
        [
            {
                "公告时间": "2026-08-29",
                "公告标题": "2026年半年度报告",
                "公告链接": "/finalpage/semi.pdf",
            },
            {
                "公告时间": "2026-04-30",
                "公告标题": "2026年第一季度报告",
                "公告链接": "/finalpage/q1.pdf",
            },
            {
                "公告时间": "2026-03-25",
                "公告标题": "2025年年度报告",
                "公告链接": "/finalpage/annual.pdf",
            },
        ]
    )
    selected = select_high_value_disclosures(
        disclosures,
        ticker="000001.SZ",
        annual_year=2025,
        interim_year=2026,
    )
    assert [item.doc_type for item in selected] == [
        "annual_report",
        "semiannual_report",
        "quarterly_report",
    ]
    assert all(item.url.startswith("https://") for item in selected)
