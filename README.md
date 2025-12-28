# my-ai-agent

开源项目新进展 AI agent.

## 快速开始

### 1) 准备环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 配置

```bash
cp config.example.yaml config.yaml
```

支持在运行时传入配置文件:

```bash
python3 agent.py --config ./config.yaml
```

### 3) 运行

例如, 生成最近 7 天 `main` 分支新增提交的摘要:

```bash
python3 agent.py --repo prometheus/prometheus --branch main --days 7 --out ./out
```
