import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from openai import OpenAI


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    candidates = [Path(path)] if path else [Path("config.yaml"), Path("config.yml")]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise RuntimeError("Config file not found. Please pass --config ./config.yaml")
    raw = p.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) if raw.strip() else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise RuntimeError("Invalid config: root must be a YAML mapping/object")
    return data


def cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="OpenAI chat completion example (reads single YAML config).")
    p.add_argument("--config", default=None, help="Path to YAML config (default: ./config.yaml or ./config.yml).")
    p.add_argument("--prompt", default="Hello!", help="User prompt.")
    args = p.parse_args(argv)

    cfg = load_yaml_config(args.config)
    api_key = cfg_get(cfg, "openai.api_key", "") or ""
    base_url = cfg_get(cfg, "openai.base_url", "https://idealab.alibaba-inc.com/api/openai/v1")
    model = cfg_get(cfg, "openai.model", "qwen3-coder-plus")
    if not api_key:
        raise RuntimeError("Missing openai.api_key in config.yaml")

    client = OpenAI(api_key=api_key, base_url=base_url)

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.prompt}],
    )

    print(completion.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())