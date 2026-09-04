"""Tiny terminal client for the v1.4 /chat API."""

from __future__ import annotations
import argparse
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    thread_id = None
    print("TradingAgents对话模式。输入 /quit 退出，/new 新建会话。")
    while True:
        message = input("你> ").strip()
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/new":
            thread_id = None
            print("已新建会话")
            continue
        payload = {"message": message, "thread_id": thread_id, "ticker": args.ticker}
        if args.date:
            payload["as_of_date"] = args.date
        response = requests.post(f"{args.url.rstrip('/')}/chat", json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        thread_id = data["thread_id"]
        print(f"助手[{data['route']}]> {data['answer']}\n")


if __name__ == "__main__":
    main()
