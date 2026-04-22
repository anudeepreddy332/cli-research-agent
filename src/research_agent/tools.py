import os
import httpx
from pathlib import Path
from bs4 import BeautifulSoup
from tavily import TavilyClient
import datetime as dt
import asyncio

REPORTS_DIR = Path("reports")
_PAGE_CACHE: dict[str, str] = {}

# Page fetching

async def _fetch_single(url: str, client: httpx.AsyncClient) -> str:
    if url in _PAGE_CACHE:
        return _PAGE_CACHE[url]

    try:
        resp = await client.get(url, timeout=12.0)
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove nav, footer, sidebar, noise
        for tag in soup(["nav", "footer", "aside", "script", "style", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)[:4000]
        if len(text) < 200:
            return f"URL: {url}\n\n[Content too short or blocked]"
        _PAGE_CACHE[url] = text
        return f"URL: {url}\n\n{text}"

    except Exception as e:
        return f"Error fetching {url}: {e}"


async def _fetch_all(urls: list[str]) -> str:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [_fetch_single(url, client) for url in urls]
        results = await asyncio.gather(*tasks)
    return "\n\n---\n\n".join(results)



def fetch_page(urls: list[str]) -> str:
    return asyncio.run(_fetch_all(urls))


# Web search
def web_search(query: str) -> str:
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily.search(query=query, search_depth="advanced", max_results=5)
    lines = []
    for i, r in enumerate(response.get("results", [])[:5], 1):
        lines.append(f"{i}. {r.get('title')}\nURL: {r.get('url')}\n{r.get('content')}\n")
    return "\n".join(lines) if lines else "No results found."


# Build calculate tool
import ast
import operator as op

_SAFE_OPS = {
    ast.Add: op.add, ast.Sub: op.sub,
    ast.Mult:op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"




# Report writing
def write_report(title: str, summary: str, key_points: list[str], sources: list[str]) -> str:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = REPORTS_DIR / f"report_{timestamp}.md"

    lines = [
        f"# {title}\n",
        f"## Summary\n{summary}\n",
        "## Key Points\n",
        *[f"- {p}\n" for p in key_points],
        "\n## Sources\n",
        *[f"- {s}\n" for s in sources],
    ]

    filename.write_text("".join(lines))
    return f"Report written to {filename}"



# Dispatcher
def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name == "web_search":
        query = arguments.get("query")
        if not query:
            return f"Error: missing 'query' parameter"
        return web_search(query)


    elif tool_name == "fetch_page":
        urls = arguments.get("urls") or ([arguments["url"]] if "url" in arguments else None)

        if not urls:
            return "Error: missing 'urls' parameter"

        return fetch_page(urls)


    elif tool_name == "write_report":
        return write_report(
            arguments.get("title", "Untitled"),
            arguments.get("summary", ""),
            arguments.get("key_points", []),
            arguments.get("sources", []),
        )


    elif tool_name == "calculate":
        expr = arguments.get("expression")
        if not expr:
            return "Error: missing 'expression' parameter"
        return calculate(expr)


    return f"Error: Unknown tool '{tool_name}'"


# ADD — tool schemas for LangGraph (same content as TOOLS in main.py):
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use when you need up-to-date or external knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch the full content of webpages given their URLs. Use after web_search to read pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs to fetch",
                    }
                },
                "required": ["urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression safely. Use when you need to compute percentages, ratios, or derive numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '294000 / 1000000'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": "Write the final research report to a markdown file. Call this once you have enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string", "description": "2-3 sentence overview"},
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key findings",
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs used",
                    },
                },
                "required": ["title", "summary", "key_points", "sources"],
            },
        },
    },
]