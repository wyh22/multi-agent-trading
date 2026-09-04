from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _run_coro_sync(coro):
    """Run an async MCP loader from sync CLI code, even if caller owns an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def worker():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001
            error["value"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error["value"]
    return result.get("value")


async def _load_servers(servers: dict[str, dict[str, Any]]):
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise RuntimeError("langchain-mcp-adapters 未安装，请执行 `pip install -e '.[agent]'`") from exc
    client = MultiServerMCPClient(servers, handle_tool_errors=False)
    return await client.get_tools()


def _to_sync_tool(remote_tool):
    """Wrap async MCP adapter tools for the existing synchronous LangGraph ToolNode."""
    from langchain_core.tools import StructuredTool

    def invoke_remote(**kwargs):
        return _run_coro_sync(remote_tool.ainvoke(kwargs))

    invoke_remote.__name__ = remote_tool.name
    return StructuredTool.from_function(
        func=invoke_remote,
        name=remote_tool.name,
        description=remote_tool.description or f"MCP tool {remote_tool.name}",
        args_schema=remote_tool.args_schema,
    )


def load_mcp_tools_sync(url: str | None = None, *, servers: dict[str, dict[str, Any]] | None = None):
    """Load one finance MCP server or an explicit multi-server MCP configuration."""
    if servers is None:
        if not url:
            raise ValueError("url或servers至少提供一个")
        servers = {"finance": {"transport": "http", "url": url}}
    remote = _run_coro_sync(_load_servers(servers))
    return [_to_sync_tool(tool) for tool in remote]
