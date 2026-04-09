import json

import openai
import time
from src.research_agent.config import get_deepseek_client
from src.research_agent.tools import execute_tool


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


SYSTEM_PROMPT = """You are a thorough research analyst. Produce comprehensive, detailed reports.
 
WORKFLOW — follow exactly:
1. Call web_search once with the user's question.
2. Pick the top 3-4 URLs from search results.
3. Call fetch_page ONCE with all URLs together: {"urls": ["url1", "url2", "url3"]}
4. Read all fetched content carefully.
5. Call write_report once with a rich, detailed report.
 
REPORT QUALITY REQUIREMENTS:
- summary: 4-6 sentences covering background, main findings, and real-world implications
- key_points: produce AT LEAST 10 distinct points. Each point must be 2-3 sentences long
  with specifics — include numbers, names, techniques, or examples where available.
  Do NOT write one-liners. Do NOT be vague.
- sources: include every URL you fetched
 
RULES:
- Never call fetch_page more than once.
- Never call write_report before fetch_page.
- After write_report, stop.
- Only call calculate when you have actual numerical findings that need derivation — not for date arithmetic."""

def run_agent(question: str):
    client = get_deepseek_client()
    fetch_called = False

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(10):
        print(f"\nStep {step + 1}:", end=" ", flush=True)

        try:
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4000,
                temperature=0.3,
                stream=True,
            )

            # Collect streamed chunks into a final message
            content_parts = []
            tool_calls_map = {} # index -> {id, name, args}

            for chunk in stream:
                delta = chunk.choices[0].delta
                finish = chunk.choices[0].finish_reason

                # Stream visible text to terminal in real time
                if delta.content:
                    print(delta.content, end="", flush=True)
                    content_parts.append(delta.content)

                # Accumulate tool call fragments
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc_chunk.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_chunk.id:
                            tool_calls_map[idx]["id"] = tc_chunk.id
                        if tc_chunk.function:
                            if tc_chunk.function.name:
                                tool_calls_map[idx]["function"]["name"] += tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                tool_calls_map[idx]["function"]["arguments"] += tc_chunk.function.arguments

                if finish:
                    final_finish = finish

            if content_parts:
                print() # new line after streamed text

            # Reconstruct into the same shape as the rest of the loop
            tool_calls_list = [tool_calls_map[i] for i in sorted(tool_calls_map)] if tool_calls_map else []

            # Fake a msg object the existing code can use

            class _Msg:
                content = "".join(content_parts) or None
                tool_calls = [
                    type("TC", (), {
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": type("F", (), {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        })()
                    })()
                    for tc in tool_calls_list
                ] or None

            class _Choice:
                message = _Msg()
                finish_reason = final_finish

            class _Response:
                choices = [_Choice()]


            response = _Response()
            choice = response.choices[0]
            msg = choice.message
            finish = choice.finish_reason

        except openai.RateLimitError:
            print("Rate limit hit - waiting 30s...")
            time.sleep(30)
            continue

        except openai.BadRequestError as e:
            print(f"Bad request: {e}")
            break

        choice = response.choices[0]
        msg = choice.message
        finish = choice.finish_reason

        if finish == "stop" or not msg.tool_calls:
            if msg.content:
                print(msg.content)
            break

        # Append the assistant message with tool_calls first
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })


        # Execute each tool and append its result immediately
        report_written = False
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            print(f" -> calling {tool_name}")

            # Guard: only call fetch_page once
            if tool_name == "fetch_page" and fetch_called:
                result = "Skipped: fetch_page already called once."
            else:
                if tool_name == "fetch_page":
                    fetch_called = True
                result = execute_tool(tool_name, args)

            print(f"     result: {len(str(result))} chars")


            # Append tool result — MUST use the same tool_call_id
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

            if tool_name == "write_report":
                report_written = True

        if report_written:
            print("\n✓ Report written. Done.")
            break


def main():
    question = input("Enter your research question: ")
    run_agent(question)



if __name__ == "__main__":
    main()