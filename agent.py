from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

try:
    from urllib3.exceptions import NotOpenSSLWarning

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    candidates = []
    if path:
        candidates = [Path(path)]
    else:
        candidates = [Path("config.yaml"), Path("config.yml")]

    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise RuntimeError("Config file not found. Please pass --config ./config.yaml")

    try:
        raw = p.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) if raw.strip() else {}
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("root must be a YAML mapping/object")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load YAML config file: {p}") from e


def cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def iso_date(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%d")

def iso_datetime(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clamp_text(s: str, max_len: int) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def char_len_zh(s: str) -> int:
    return len(re.sub(r"\s+", "", s))

def _normalize_zh_paragraph(text: str, *, target: int = 250, min_len: int = 220, max_len: int = 280) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    fillers = [
        "从提交信息与合并记录看，近期工作以修复与维护为主，同时穿插少量功能演进。",
        "建议结合变更范围与影响面，优先回看涉及核心路径与接口变更的提交，并补充回归验证。",
        "如果后续要继续追踪，可按模块/作者/PR主题分组，对重复出现的问题建立专项清单。",
    ]
    i = 0
    while char_len_zh(s) < min_len and i < len(fillers):
        s = (s + " " + fillers[i]).strip()
        i += 1
    if char_len_zh(s) > max_len:
        s = re.sub(r"\s+", " ", s)[: max_len]
        s = re.sub(r"[，,、。\.]+?$", "", s).strip() + "。"
    return s

@dataclass
class CommitDigest:
    sha: str
    short_sha: str
    message: str
    url: str
    authored_at: str
    author: str


class GitHubClient:
    def __init__(self, token: Optional[str]) -> None:
        self.session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "my-ai-agent/1.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)

    def _request_json(self, method: str, url: str, params: Dict[str, Any]) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(1, 5):
            try:
                resp = self.session.request(method, url, params=params, timeout=30)
                if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                    reset = resp.headers.get("X-RateLimit-Reset")
                    sleep_s = 10
                    if reset and reset.isdigit():
                        sleep_s = max(1, int(reset) - int(time.time()))
                    time.sleep(min(sleep_s, 60))
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"GitHub API request failed after retries: {url}") from last_err

    def list_commits_since(
        self,
        repo: str,
        branch: str,
        since_utc: datetime,
        until_utc: datetime,
        max_commits: int,
    ) -> List[Dict[str, Any]]:
        url = f"https://api.github.com/repos/{repo}/commits"
        items: List[Dict[str, Any]] = []
        page = 1
        per_page = 100
        while len(items) < max_commits:
            params = {
                "sha": branch,
                "since": iso_datetime(since_utc),
                "until": iso_datetime(until_utc),
                "per_page": per_page,
                "page": page,
            }
            data = self._request_json("GET", url, params=params)
            batch = data if isinstance(data, list) else []
            if not batch:
                break
            items.extend([it for it in batch if isinstance(it, dict)])
            if len(batch) < per_page:
                break
            page += 1
        return items[: max(1, int(max_commits))]

def parse_datetime(s: str) -> datetime:
    if not s:
        return utc_now()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return utc_now()


def categorize_commit(message: str) -> str:
    m = (message or "").strip().lower()
    first = m.splitlines()[0] if m else ""
    if any(k in first for k in ["docs:", "doc:", "readme", "documentation"]):
        return "文档"
    if any(k in first for k in ["build:", "ci:", "test:", "lint:", "chore:"]):
        return "工程/CI"
    if any(k in first for k in ["fix:", "bug", "regression", "panic", "crash"]):
        return "Bug/修复"
    if any(k in first for k in ["perf:", "performance"]):
        return "性能"
    if any(k in first for k in ["ui", "web"]):
        return "UI/Web"
    if any(k in first for k in ["feat:", "feature", "enhancement"]):
        return "需求/改进"
    return "其他"


def group_by_category_commits(digests: List[CommitDigest]) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for d in digests:
        c = categorize_commit(d.message)
        counts[c] = counts.get(c, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))


def draft_commits_report_heuristic(
    repo: str,
    branch: str,
    start_utc: datetime,
    end_utc: datetime,
    digests: List[CommitDigest],
    days: int,
) -> str:
    total = len(digests)
    authors = sorted({d.author for d in digests if d.author})
    cat_counts = group_by_category_commits(digests)[:6]
    cat_line = "；".join([f"{c}{n}条" for c, n in cat_counts]) if cat_counts else "（无）"

    merges = sum(1 for d in digests if (d.message or "").lower().startswith("merge "))
    deps = sum(1 for d in digests if "deps" in (d.message or "").lower())
    fixes = sum(1 for d in digests if any(k in (d.message or "").lower() for k in ["fix", "bugfix", "bug"]))
    feats = sum(1 for d in digests if any(k in (d.message or "").lower() for k in ["feat", "feature", "add "]))

    highlights = digests[: min(10, total)]
    hl_lines = "\n".join(
        [
            f"- `{d.short_sha}` {clamp_text(d.message.splitlines()[0], 90)}（{d.author} / {d.authored_at}）"
            for d in highlights
        ]
    )

    p1 = _normalize_zh_paragraph(
        f"最近{days}天（{iso_date(start_utc)}~{iso_date(end_utc)} UTC）{repo} 的 `{branch}` 分支新增提交 {total} 条，涉及作者 {len(authors)} 位。"
        f"其中合并类提交约 {merges} 条，依赖/自动化更新约 {deps} 条；从提交标题关键词看，修复相关约 {fixes} 条、功能增强相关约 {feats} 条，整体节奏偏向稳定性与维护。",
        target=250,
    )
    p2 = _normalize_zh_paragraph(
        f"按主题粗分布来看：{cat_line}。这通常意味着近期变更多围绕工程维护（依赖/CI）与局部问题修正推进，"
        f"同时夹杂少量面向用户的能力增强与体验改进。建议优先关注“Bug/修复”与核心链路相关的提交，"
        f"并将较大的改动拆解到具体模块进行复核，避免小改动累计造成行为变化。",
        target=250,
    )
    p3 = _normalize_zh_paragraph(
        "从落地策略上，建议对合并提交先回溯对应 PR 说明与关联讨论，确认动机、影响面与回滚路径；"
        "对依赖更新类提交，建议核对安全公告、许可证变化与编译/运行时兼容性，并补充最小集成测试。"
        "如果要进一步提高可读性，可在后续报告中把提交按模块聚合、挑选 10-20 条“最具代表性变更”单独展开说明。",
        target=250,
    )
    md = (
        f"## {repo} 最近{days}天新增提交摘要（{branch}）\n\n"
        f"- 时间范围：{iso_date(start_utc)} ~ {iso_date(end_utc)}（UTC）\n"
        f"- 分支：`{branch}`\n"
        f"- 新增提交数：{total}\n"
        f"- 作者数：{len(authors)}\n\n"
        f"主题分布：{cat_line}\n\n"
        f"{p1}\n\n{p2}\n\n{p3}\n\n"
        f"新增提交列表（按时间倒序，最多展示10条）：\n{hl_lines}\n"
    )
    return md


def write_outputs(out_dir: Path, digests: List[CommitDigest], report_md: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    commits_path = out_dir / "commits.json"
    report_path = out_dir / "weekly_report.md"

    commits_path.write_text(
        json.dumps([d.__dict__ for d in digests], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(report_md, encoding="utf-8")

def openai_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package is not available. Please `pip install -r requirements.txt`.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    try:
        return resp.model_dump()
    except Exception:
        return json.loads(resp.model_dump_json())


def to_commit_digest(item: Dict[str, Any]) -> CommitDigest:
    sha = (item.get("sha") or "").strip()
    short_sha = sha[:7] if sha else ""
    html_url = item.get("html_url") or ""
    commit = item.get("commit") or {}
    message = (commit.get("message") or "").strip()
    authored_at = ((commit.get("author") or {}).get("date")) or ""
    author = (
        ((item.get("author") or {}).get("login"))
        or ((commit.get("author") or {}).get("name"))
        or "unknown"
    )
    return CommitDigest(
        sha=sha,
        short_sha=short_sha,
        message=message,
        url=html_url,
        authored_at=authored_at,
        author=author,
    )


def tool_get_new_commits(
    *,
    repo: str,
    branch: str,
    days: int,
    github_token: Optional[str],
    max_commits: int,
) -> Dict[str, Any]:
    end_utc = utc_now()
    start_utc = end_utc - timedelta(days=max(1, int(days)))

    gh = GitHubClient(token=github_token)
    items = gh.list_commits_since(
        repo=repo,
        branch=branch,
        since_utc=start_utc,
        until_utc=end_utc,
        max_commits=max(1, int(max_commits)),
    )
    digests = [to_commit_digest(it) for it in items]

    digests_sorted = sorted(digests, key=lambda d: parse_datetime(d.authored_at), reverse=True)
    return {
        "repo": repo,
        "branch": branch,
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "total": len(digests_sorted),
        "commits": [d.__dict__ for d in digests_sorted],
    }


def draft_commits_report_openai_agent(
    *,
    repo: str,
    branch: str,
    days: int,
    max_commits: int,
    github_token: Optional[str],
    openai_api_key: str,
    openai_base_url: str,
    openai_model: str,
) -> Tuple[str, List[CommitDigest]]:
    payload = tool_get_new_commits(
        repo=repo,
        branch=branch,
        days=days,
        github_token=github_token,
        max_commits=max_commits,
    )

    digests: List[CommitDigest] = []
    if isinstance(payload.get("commits"), list):
        digests = [CommitDigest(**d) for d in payload["commits"]]

    compact_commits = []
    for d in digests[: min(len(digests), 400)]:
        title = clamp_text((d.message or "").splitlines()[0], 120)
        compact_commits.append(
            {
                "short_sha": d.short_sha,
                "title": title,
                "author": d.author,
                "authored_at": d.authored_at,
                "url": d.url,
            }
        )
    payload_for_llm = {
        "repo": payload.get("repo"),
        "branch": payload.get("branch"),
        "start_utc": payload.get("start_utc"),
        "end_utc": payload.get("end_utc"),
        "total": payload.get("total"),
        "commits": compact_commits,
        "note": (
            "commits 列表为压缩视图（仅 short_sha + title 等）。如 total 远大于列表长度，请基于现有列表与统计做概括。"
        ),
    }

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是一个开源项目维护者助理。你会根据给定的 issue 摘要数据，生成中文周报。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"请基于以下 JSON 数据，为目标仓库生成“仅最近{days}天 main 分支新增提交”的摘要。\n"
                "要求：\n"
                f"- 文中不要使用“本周”，统一使用“最近{days}天”表述\n"
                f"- 开头一句以“最近{days}天”开头，给出新增提交数量概览\n"
                "- 输出必须包含：\n"
                "  - 新增提交列表（至少10条或全部，如果不足10条就全部；每条包含 short_sha + 一句话概括）\n"
                "  - 总体总结（必须 3 段正文；每段约250字，建议控制在 220~280 字之间；段落之间空一行）\n"
                "    - 第1段：整体概览（数量、主要类型、节奏）\n"
                "    - 第2段：主题/模块趋势（从提交标题归纳，指出 2-4 个主要方向）\n"
                "    - 第3段：风险点与建议（回归验证、关注点、后续跟踪建议）\n"
                "- 输出为 Markdown\n\n"
                f"参数：repo={repo}, branch={branch}, days={days}, max_commits={max_commits}\n\n"
                "数据（JSON）：\n"
                f"{json.dumps(payload_for_llm, ensure_ascii=False)}"
            ),
        },
    ]

    data = openai_chat_completions(
        base_url=openai_base_url,
        api_key=openai_api_key,
        model=openai_model,
        messages=messages,
        temperature=0.2,
    )
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        return "", digests

    md = (
        f"## {repo} 最近{days}天新增提交摘要（{branch}）\n\n"
        f"- 时间范围：最近{days}天（UTC）\n"
        f"- 分支：`{branch}`\n"
        f"- 数据：GitHub Commits\n\n"
        f"{content}\n"
    )
    return md, digests


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AI Agent: summarize last-N-days new GitHub commits on a branch.")
    p.add_argument(
        "--config",
        default=None,
        help="Path to a single YAML config file (default: auto-load ./config.yaml or ./config.yml).",
    )
    p.add_argument("--repo", default=None, help="Override: GitHub repo in owner/name form.")
    p.add_argument("--days", type=int, default=None, help="Override: only include issues created in last N days.")
    p.add_argument("--branch", default=None, help="Override: branch name (default: main).")
    p.add_argument("--max-commits", type=int, default=None, help="Override: max commits to process.")
    p.add_argument("--state", choices=["open", "all"], default=None, help=argparse.SUPPRESS)
    p.add_argument("--max-issues", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--out", default=None, help="Override: output directory.")
    p.add_argument("--mode", choices=["auto", "heuristic", "openai-agent"], default=None, help="Override run mode.")
    args = p.parse_args(argv)

    cfg = load_yaml_config(args.config)

    repo = args.repo or cfg_get(cfg, "github.repo", "prometheus/prometheus")
    days = int(args.days) if args.days is not None else int(cfg_get(cfg, "run.days", 7))
    branch = args.branch or cfg_get(cfg, "run.branch", "main")
    max_commits = (
        int(args.max_commits)
        if args.max_commits is not None
        else int(cfg_get(cfg, "run.max_commits", cfg_get(cfg, "run.max_issues", 200)))
    )
    if args.max_commits is None and args.max_issues is not None:
        max_commits = int(args.max_issues)
    out_dir = args.out or cfg_get(cfg, "run.out", "./out")
    mode = args.mode or cfg_get(cfg, "run.mode", "auto")

    github_token = cfg_get(cfg, "github.token", None)
    openai_api_key = cfg_get(cfg, "openai.api_key", None)
    openai_base_url = cfg_get(cfg, "openai.base_url", "https://api.openai.com/v1")
    openai_model = cfg_get(cfg, "openai.model", "gpt-4o-mini")

    use_openai_agent = mode == "openai-agent" or (mode == "auto" and bool(openai_api_key))
    if mode == "openai-agent" and not openai_api_key:
        raise RuntimeError("mode=openai-agent requires openai.api_key in config.yaml")

    if use_openai_agent and openai_api_key:
        md, digests_from_tool = draft_commits_report_openai_agent(
            repo=repo,
            branch=branch,
            days=days,
            max_commits=max_commits,
            github_token=github_token,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
        )
        if md and digests_from_tool:
            write_outputs(Path(out_dir), digests_from_tool, md)
            return 0
        pass

    end_utc = utc_now()
    start_utc = end_utc - timedelta(days=max(1, days))
    gh = GitHubClient(token=github_token)
    items = gh.list_commits_since(
        repo=repo,
        branch=branch,
        since_utc=start_utc,
        until_utc=end_utc,
        max_commits=max(1, max_commits),
    )
    digests = [to_commit_digest(it) for it in items]
    digests_sorted = sorted(digests, key=lambda d: parse_datetime(d.authored_at), reverse=True)
    report = draft_commits_report_heuristic(repo, branch, start_utc, end_utc, digests_sorted, days=days)
    write_outputs(Path(out_dir), digests_sorted, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


