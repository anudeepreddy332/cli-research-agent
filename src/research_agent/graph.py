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
import os
from typing import Literal, Any
from typing_extensions import TypedDict
from src.research_agent.config import (DEEPSEEK_MODEL, DEEPSEEK_BASE_URL,
                                       COST_PER_1M_INPUT, COST_PER_1M_OUTPUT,
                                       MAX_ITERATIONS, MAX_COST_PER_RUN)
from langchain_openai import ChatOpenAI
from src.research_agent.tools import execute_tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.research_agent.tools import TOOL_SCHEMAS
from dotenv import load_dotenv
load_dotenv()

TOOLS = TOOL_SCHEMAS


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
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=4000,
    )

# Helper: track cost and tokens from LLM response
def _track_cost(state: AgentState, response: Any) -> dict:
    """
    Extract token usage from LangChain response and compute accumulated cost.
    Returns dict with new running totals to merge into state.
    LangGraph state merge is additive by replacement (not addition) —
    the returned values must be the new totals, not deltas.
    """
    usage = response.usage_metadata
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = (input_tokens / 1e6 * COST_PER_1M_INPUT) + (output_tokens / 1e6 * COST_PER_1M_OUTPUT)
    return {
        "total_cost_usd": state["total_cost_usd"] + cost,
        "total_tokens": state["total_tokens"] + input_tokens + output_tokens,
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

    client_with_tools = client.bind_tools(lc_tools, tool_choice="auto")
    response = client_with_tools.invoke(lc_messages)
    cost_update = _track_cost(state, response)

    # Build assistant message dict (OpenAI format)
    assistant_msg = {
        "role": "assistant",
        "content": response.content or "",
    }
    if response.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc["id"],
                "name": tc["name"],
                "args": tc["args"],
            }
            for tc in response.tool_calls
        ]

    return {
        "messages": state["messages"] + [assistant_msg],
        "iterations": state["iterations"] + 1,
        **cost_update,
    }

def node_tool_executor(state: AgentState) -> dict:
    """
    Execute tool calls from the last assistant message.
    Appends tool result messages and updates fetch_called/report_written flags.
    """
    last_msg = state["messages"][-1]
    tool_calls = last_msg.get("tool_calls", [])
    if not tool_calls:
        return {}

    new_messages = []
    fetch_called = state["fetch_called"]
    report_written = state["report_written"]

    for tc in tool_calls:
        if "function" in tc:
            # OpenAI nested format
            tool_name = tc["function"]["name"]
            args_str = tc["function"]["arguments"]
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

        else:
            tool_name = tc["name"]
            args = tc["args"]


        # Enforce fetch_page once
        if tool_name == "fetch_page" and fetch_called:
            result = "Skipped: fetch_page already called once."

        else:
            if tool_name == "fetch_page":
                fetch_called = True
            result = execute_tool(tool_name, args)
            if tool_name == "write_report":
                report_written = True

        new_messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": str(result),
        })

    return {
        "messages": state["messages"] + new_messages,
        "fetch_called": fetch_called,
        "report_written": report_written,
        "status": "done" if report_written else "running",
    }


# Conditional edge: should_continue

def should_continue(state: AgentState) -> Literal["tool_executor", "end"]:
    """
        Central routing logic. Called after call_model.
        - If cost exceeded -> end
        - If max iterations reached -> end
        - If report written -> end
        - If last message has tool calls -> tool_executor
        - Else -> end (no tool calls, assistant gave final answer but didn't call write_report? Actually prompt forces write_report, but guard here)
    """
    if state["total_cost_usd"] > MAX_COST_PER_RUN:
        print(f"[gate] Cost exceeded: ${state['total_cost_usd']:.4f}")
        return "end"

    if state["iterations"] >= MAX_ITERATIONS:
        print(f"[gate] Max iterations ({MAX_ITERATIONS}) reached")
        return "end"

    if state.get("report_written"):
        print("[gate] Report written, ending")
        return "end"

    last_msg = state["messages"][-1]
    if last_msg.get("tool_calls"):
        return "tool_executor"
    return "end"


# Graph assembly
def build_graph():
    """
    Construct and compile the LangGraph state machine.
    Returns a runnable graph object.
    """
    builder = StateGraph(AgentState)

    builder.add_node("call_model", node_call_model)
    builder.add_node("tool_executor", node_tool_executor)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "tool_executor": "tool_executor",
            "end": END,
        }
    )
    builder.add_edge("tool_executor", "call_model")     # Loop back

    return builder.compile()
