from __future__ import annotations
import argparse


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    p = argparse.ArgumentParser(description="扩展版 CLI")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--name", type=str, default="world")
    p.add_argument("--verbose", action="store_true")
    return p


def parse(argv: list[str]) -> dict:
    """解析命令行参数并返回字典"""
    a = build_parser().parse_args(argv)
    return {"n": a.n, "name": a.name, "verbose": a.verbose}
