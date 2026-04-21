"""
LangGraph implementation of CLI Research Agent

State machine:
    Start -> call_model (LLM) -> should_continue -> tool_executor -> call_model (loop)
                                                 -> End (if no tool calls or write_report called)

Nodes:
    - call_model: Calls Deepseek with current messages, returns assistant response.
    - tool_executor: Executes any tool calls in the last assistant message.
"""
import json
import time
from typing import Literal, Any
from typing_extensions import TypedDict
from src.research_agent.config_langgraph import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, \
    COST_PER_1M_INPUT_TOKENS, COST_PER_1M_OUTPUT_TOKENS
from langchain_openai import ChatOpenAI
from src.research_agent.tools import execute_tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information."
                           "Use when you need up-to-date or external knowledge.",
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
            "description": "Fetch the full content of webpages given their URLs."
                           "Use after web_search to read pages.",
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
            "description": "Evaluate a math expression safely."
                           "Use when you need to compute percentages, ratios, or derive numbers from research findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '294000 / 1000000' or '41 / 100 * 54'",
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
            "description": "Write the final research report to a markdown file."
                           "Call this once you have enough information.",
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


SYSTEM_PROMPT = """You are a research analyst. Always deliver output as a written report using write_report — never respond with plain text.

DECIDE FIRST: Does this question require current, live, or external information (news, prices, recent events, specific URLs, statistics that change)?

IF YES — follow this workflow exactly:
1. Call web_search once.
2. Pick the top 3-4 URLs from results.
3. Call fetch_page ONCE with all URLs together.
4. Optionally call calculate if you need to derive numbers.
5. Call write_report with detailed findings.

IF NO (you can answer fully from your training knowledge) — skip directly to:
1. Call write_report immediately using what you know.
   Set sources to ["internal knowledge"] if no URLs were used.

REPORT QUALITY REQUIREMENTS — apply regardless of path:
- summary: 4-6 sentences covering background, main findings, and real-world implications.
- key_points: AT LEAST 10 distinct points, each 2-3 sentences with specifics, names, numbers, or examples. No one-liners.

RULES:
- Never respond with plain text. Always call write_report as your final action.
- Never call fetch_page more than once.
- Never call write_report before fetch_page if you chose the web search path.
- Only call calculate when you have numerical findings that need derivation. Not for date arithmetic."""


class AgentState(TypedDict):
    """
    All fields that any node reads or writes must be declared here.
    LangGraph uses this to validate state at compile time.
    """
    messages: list[dict]
    question: str
    fetch_called: bool
    report_written: bool
    iterations: int
    total_cost_usd: float
    total_tokens: int
    status: Literal["running", "done", "blocked", "cost_exceeded"]

# Helper: create DeepSeek client via LangChain
def _make_client() -> ChatOpenAI:
    """
    Build ChatOpenAI client pointed at DeepSeek.
    Why LangChain? LangGraph nodes work naturally with LangChain message objects.
    """
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=4000,
    )

# Helper: track cost and tokens from LLM response
def _track_cost(state: AgentState, response: Any) -> dict:
    usage = response.usage_metadata
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = (input_tokens / 1e6 * COST_PER_1M_INPUT_TOKENS) + (output_tokens / 1e6 * COST_PER_1M_OUTPUT_TOKENS)
    return {
        "total_cost_usd": state["total_cost_usd"] + cost
        "total_tokens": state["total_tokens"] + input_tokens + output_tokens
    }

# Node 1: call_model
def node_call_model(state: AgentState) -> dict:
    """
    Call DeepSeek with the current message history.
    Appends assistant message (with optional tool_calls) to state.messages.
    Updates iterations and cost.
    """
    client = _make_client()

    # Convert dict messages to LangChain format
    lc_messages = []
    for m in state["messages"]:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            ai_msg = AIMessage(content=content)
            if "tool_calls" in m:
                ai_msg.tool_calls = m["tool_calls"]
            lc_messages.append(ai_msg)
        elif role == "tool":
            lc_messages.append(ToolMessage(
                content=content,
                tool_call_id=m.get("tool_call_id", ""),
            ))

    # Convert TOOLS to LangChain format
    lc_tools = [
        {
            "type": "function",
            "function": {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "parameters": t["function"]["parameters"],
            },
        }
        for t in TOOLS
    ]

    response = client.invoke(lc_messages, tools=lc_tools, tool_choice="auto")
    cost_update = _track_cost(state, response)















