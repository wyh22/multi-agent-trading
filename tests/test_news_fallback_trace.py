from pathlib import Path

import pandas as pd

import tradingagents.dataflows.akshare_news as akshare_news
import tradingagents.dataflows.interface as data_interface


ROOT = Path(__file__).resolve().parents[1]


class FakeAkshare:
    @staticmethod
    def stock_news_em(symbol: str):
        assert symbol == "601016"
        return pd.DataFrame(
            [
                {
                    "关键词": "601016",
                    "新闻标题": "窗口内新闻",
                    "新闻内容": "公司近期经营信息。",
                    "发布时间": "2026-09-04 10:00:00",
                    "文章来源": "东方财富",
                    "新闻链接": "https://example.com/in-window",
                },
                {
                    "关键词": "601016",
                    "新闻标题": "未来新闻",
                    "新闻内容": "不得进入 PIT 结果。",
                    "发布时间": "2026-09-08 10:00:00",
                    "文章来源": "东方财富",
                    "新闻链接": "https://example.com/future",
                },
            ]
        )


def test_eastmoney_company_news_fallback_is_pit_filtered(monkeypatch):
    monkeypatch.setattr(
        akshare_news,
        "_check_akshare",
        lambda: FakeAkshare(),
    )
    result = akshare_news.get_akshare_company_news(
        "601016.SH",
        "2026-08-01",
        "2026-09-06",
    )
    assert "eastmoney-stock-news" in result
    assert "窗口内新闻" in result
    assert "未来新闻" not in result
    assert "不替代交易所/巨潮资讯法定公告原文" in result


def test_get_news_falls_back_from_cninfo_to_akshare(monkeypatch):
    def cninfo_fail(*_args, **_kwargs):
        raise ValueError("cninfo non-json")

    def eastmoney_ok(*_args, **_kwargs):
        return "EASTMONEY_BACKUP_OK"

    monkeypatch.setattr(
        data_interface,
        "get_vendor",
        lambda _category, _method=None: "cninfo,akshare",
    )
    monkeypatch.setitem(
        data_interface.VENDOR_METHODS,
        "get_news",
        {
            "cninfo": cninfo_fail,
            "akshare": eastmoney_ok,
        },
    )
    assert (
        data_interface.route_to_vendor(
            "get_news",
            "601016.SH",
            "2026-08-01",
            "2026-09-06",
        )
        == "EASTMONEY_BACKUP_OK"
    )


def test_chat_response_exposes_supervisor_trace_without_benchmark_capture():
    source = (
        ROOT / "tradingagents" / "conversation" / "agent.py"
    ).read_text(encoding="utf-8")
    return_marker = '"completion_ratio": completion.completion_ratio,'
    idx = source.rfind(return_marker)
    assert idx >= 0
    tail = source[idx : idx + 1600]
    assert '"supervisor_trace": supervisor_trace,' in tail
    assert '"evaluation_evidence"' in tail


def test_web_ui_renders_complete_supervisor_trace():
    source = (
        ROOT / "service" / "static" / "index.html"
    ).read_text(encoding="utf-8")
    assert "Supervisor Trace" in source
    assert "renderSupervisorTrace" in source
    assert "path=" in source
    for column in (
        "Step",
        "Action",
        "Target",
        "Route",
        "Completion",
        "Missing",
    ):
        assert column in source
