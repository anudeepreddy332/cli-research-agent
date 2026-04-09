# CLI Research Agent

A terminal-based AI research agent that takes a question, searches the web, reads source pages, and writes a structured markdown report — all from the command line.

## What it does

You ask a question. The agent searches the web via Tavily, fetches and reads the top pages, optionally runs calculations on numerical findings, and writes a detailed report to the `reports/` folder. You watch it reason step by step in real time.

## Stack

- [DeepSeek](https://platform.deepseek.com/) — LLM backbone via OpenAI-compatible API
- [Tavily](https://tavily.com/) — search API built for LLM agents
- [httpx](https://www.python-httpx.org/) + [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) — async page fetching and HTML parsing
- [uv](https://github.com/astral-sh/uv) — fast Python package manager

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/anudeepreddy332/cli-research-agent.git
cd cli-research-agent
```

**2. Install dependencies**
```bash
uv sync
```

**3. Add your API keys**

Create a `.env` file at the project root:

    DEEPSEEK_API_KEY=your_key_here
    TAVILY_API_KEY=your_key_here

Get keys from [DeepSeek Platform](https://platform.deepseek.com/) and [Tavily](https://tavily.com/). Both have free or low-cost tiers.

**4. Run the agent**
```bash
uv run python main.py
```

You will be prompted to enter your research question. The agent runs, streams its reasoning to the terminal, and writes a markdown report to `reports/`.

**5. Read your report**
```bash
ls reports/
open reports/report_YYYYMMDD_HHMMSS.md
```

## Run benchmarks

```bash
uv run python tests/benchmark.py
```

Runs 10 diverse research questions, times each run, and saves results to `tests/benchmark_results.json`.

## Project structure

    cli-research-agent/
    ├── main.py                  # Agent loop and entry point
    ├── src/research_agent/
    │   ├── config.py            # API client setup
    │   └── tools.py             # Tool implementations (search, fetch, calculate, report)
    ├── tests/
    │   ├── benchmark.py         # Batch evaluation script
    │   └── benchmark_results.json
    └── reports/                 # Generated markdown reports

## Concepts explored

- LLM tool calling from scratch using the OpenAI-compatible messages API
- Agent loop: API call → parse tool_use → execute → inject result → repeat
- Real-time response streaming
- Async parallel page fetching with httpx
- Safe expression evaluation using Python's AST parser
- Prompt engineering for structured, detailed output
- Benchmarking agent quality across diverse query types