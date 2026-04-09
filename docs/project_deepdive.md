# CLI Research Agent — Project Deep Dive

## Project Overview

A CLI research agent is a program where a language model drives a multi-step research workflow autonomously. You give it a question. It decides what to search for, which pages to read, what numbers to verify, and how to structure the final output, without you directing each step.

This is different from asking an AI chatbot a question. A chatbot generates an answer from training data. This agent actively goes out, reads current sources, and synthesises what it finds. The output is traceable to real URLs. The reasoning happens in steps you can watch.

The project was built to understand how agents actually work at the mechanical level, not through a framework that hides the implementation, but through direct API calls where every part of the loop is visible and controllable.

---

## The full pipeline

Here is what happens from the moment you press Enter to the moment a report exists on disk.

```
You type a question
        ↓
main.py sends it to DeepSeek API with a system prompt and tool schemas
        ↓
DeepSeek returns a tool_use response: "call web_search with this query"
        ↓
main.py calls execute_tool("web_search", {"query": "..."})
        ↓
tools.py calls Tavily API → returns titles, URLs, and content snippets for top 5 results
        ↓
Result is injected back into the messages array as a tool result
        ↓
DeepSeek reads the search results, picks 3-4 URLs, returns: "call fetch_page with these URLs"
        ↓
tools.py fetches all URLs in parallel using async httpx
HTML is parsed with BeautifulSoup, noise tags removed, up to 4000 chars extracted per page
        ↓
Full page content injected into messages array
        ↓
DeepSeek reads page content, may call calculate for numerical derivations
        ↓
DeepSeek calls write_report with title, summary, key_points, sources
        ↓
tools.py writes a markdown file to reports/ with a timestamp filename
        ↓
Agent loop terminates. Report exists on disk.
```

Every arrow in this diagram is one iteration of the agent loop in `main.py`. The loop runs until the model stops calling tools or until it calls `write_report`.

---

## System Architecture

The project is split into three layers, each with a clear responsibility.

### Layer 1: The agent loop (`main.py`)

This is the control layer. It owns the conversation state, the messages array that grows with each tool call and result. It calls the API, reads the response, decides what to do next, and drives the loop forward.

The core logic is about 30 lines of Python inside a `for` loop. It checks `finish_reason` from the API response. If it is `stop`, the model is done. If it contains tool calls, the loop executes each one, appends the results, and calls the API again. That repeating cycle is the entire agent mechanism.

Nothing in this layer does any actual work. It only orchestrates.

### Layer 2: Tool implementations (`src/research_agent/tools.py`)

This is the execution layer. It receives a tool name and arguments from the loop and does the actual work: hitting the Tavily API, fetching web pages, evaluating math, writing files.

Each tool is an isolated function. The dispatcher `execute_tool` routes by name. This separation means you can improve any individual tool without touching the loop logic. You can add new tools without modifying anything except this file and the tool schema in `main.py`.

### Layer 3: Configuration (`src/research_agent/config.py`)

Handles client initialisation and environment variables. Keeps credentials out of application logic.

---

## Design Rationale

### The messages array as state

The OpenAI-compatible API is stateless. It remembers nothing between calls. Everything the model knows about the current task, the original question, search results, page content, its own reasoning, lives in the messages array that you build and pass with every request.

This makes the messages array the most important data structure in the whole system. Growing it correctly (assistant message before tool results, tool_call_id matching, no orphaned messages) is what keeps the API from throwing 400 errors. Trimming it incorrectly is what causes context loss and goal drift.

### Tool schemas as behavior contracts

The tool schemas in `TOOLS` are not just type definitions. They are part of the prompt. The model reads the `description` field to decide when to call a tool and reads the parameter descriptions to decide how to call it. A vague description produces inconsistent tool selection. A precise description with "use when X, do not use when Y" collapses the probability distribution toward correct usage.

This is why the `write_report` schema explicitly says "each key point must be 2-3 sentences with specifics." The model treats schema descriptions as instructions, not just type hints.

### Async page fetching

Pages are fetched in parallel using `asyncio.gather`. Without this, fetching four URLs sequentially at 1-2 seconds each would add 4-8 seconds to every run. With parallel fetching, all four complete in roughly the time the slowest one takes.

The `BeautifulSoup` parsing strips `nav`, `footer`, `aside`, `script`, and `style` tags before extracting text. Without this, the extracted content is dominated by navigation menus, cookie notices, and sidebar links, noise that pollutes the model's context and reduces report quality.

### Safe expression evaluation

The agent can generate any string as tool arguments. If `calculate` used Python's built-in `eval()`, a compromised or hallucinated expression like `__import__('os').system('rm -rf /')` would execute silently. The AST-based evaluator parses the expression into a syntax tree and walks it node by node, only allowing numeric literals and five arithmetic operators. Everything else raises an error before execution. This is a minimal sandbox, enough protection for arithmetic in a research agent.

### Streaming

The API supports returning tokens as they are generated rather than waiting for the complete response. Streaming does not change the final output. It changes when you see it. For an agent that might spend 15 seconds reasoning before calling a tool, streaming makes the difference between a terminal that appears frozen and one that shows you the model's thinking as it happens.

Streaming also has a practical debugging benefit: you can see if the model is going off-track mid-thought and kill the process before it completes a bad tool call.

---

## Challenges and limitations

### Blocked page content

Many URLs return JavaScript-rendered pages or actively block automated requests. `httpx` cannot execute JavaScript, it fetches raw HTML. If a page requires JavaScript to display its content, BeautifulSoup extracts almost nothing. The current fallback is a `[Content too short or blocked]` string, which the model receives and works around using only the Tavily snippet. This degrades report quality on those topics. The fix is either a JavaScript-capable fetcher (Playwright or Playwright-stealth) or retrying with alternative URLs from the search results.

### Context window as a constraint

Every tool result gets appended to the messages array. A run that fetches four pages at 4,000 chars each adds roughly 16,000 characters — about 5,000 tokens — to the context before the model writes the report. As the context grows, older content (like the original question) receives lower attention weight from the model. On long runs, the model can lose track of the original goal and optimise for whatever is most recent. The current system has no rolling summarisation or context compression.

### The model cannot verify facts

Everything in the report comes from what the fetched pages say. If a page contains an error, the model will repeat it confidently. The agent has no cross-referencing logic, it does not check whether facts from different sources agree. A citation tells you where a claim came from, not whether it is true.

### Single-pass research

The agent searches once, fetches once, and writes. If the search results are poor for a given query, the report is poor. There is no loop that evaluates the fetched content and decides to search again with a refined query. This is a fundamental architectural limitation of the current design.

### Latency

A complete run takes 40-70 seconds. Most of this is model API latency on the DeepSeek side, not local computation. Page fetching adds 2-5 seconds. There is no caching of results between runs — running the same question twice triggers the full pipeline again.

---

## Key Takeaways

Building this from scratch without a framework makes explicit something frameworks hide: an agent is a loop. The model does not autonomously run. Your code calls it, reads its output, does work, and calls it again. The intelligence is the model's; the control flow is yours.

The failure modes are also clearer without a framework: a mismatched `tool_call_id` throws a 400. A trimmed message array causes goal drift. A vague tool description causes wrong tool selection. These are not abstract concerns — they are bugs you encounter within the first few runs and have to trace through the messages array to fix.

That debugging process — reading the messages array, understanding why a particular token sequence produced a particular tool call — is the foundation of understanding how agents actually work.