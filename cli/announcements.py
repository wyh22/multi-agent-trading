"""CLI 公告兼容层。

项目运行不依赖远程公告服务；保留这两个函数是为了兼容上游 CLI 入口，
并避免在启动投研命令时产生无关网络请求。
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel


def fetch_announcements(url: str | None = None, timeout: float | None = None) -> dict:
    """返回空公告集合；参数仅用于保持上游函数签名兼容。"""
    del url, timeout
    return {"announcements": [], "require_attention": False}


def display_announcements(console: Console, data: dict) -> None:
    """仅在调用方显式传入公告内容时展示。"""
    announcements = data.get("announcements", []) if isinstance(data, dict) else []
    if not announcements:
        return
    console.print(
        Panel(
            "\n".join(str(item) for item in announcements),
            border_style="cyan",
            padding=(1, 2),
            title="Announcements",
        )
    )
