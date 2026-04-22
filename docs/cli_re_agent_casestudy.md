# Phase 1 Case Study: Building a CLI Research Agent from First Principles

## 1. Overview & Context

Phase 1 builds a terminal-based research agent that takes a question, searches the web, reads pages, performs calculations, and writes a structured markdown report.

Unlike chatbots, this system actively gathers external information and produces traceable outputs from real URLs.

This phase exists to solve one core problem:  
**How do you turn a stateless LLM into a multi-step system that can act, observe, and iterate?**

### Architecture

User Query  
→ Agent Loop (main.py)  
→ DeepSeek API  
→ Tool Call?  
→ Execute Tool (tools.py)  
→ Inject Result into Messages  
→ Repeat Loop  
→ write_report → Markdown file  

---

## 2. Core Mechanics

At its core, the system is:

**state + loop + tools**

- State = messages array  
- Loop = repeated API calls  
- Tools = external actions  

Plain English:  
You’re building a feedback loop where the model keeps asking for actions until it’s satisfied.

### Assumptions
- Model outputs valid JSON tool calls  
- Web pages are readable  
- Context window can hold all data  

### Failure Modes
- Tool mismatch → API errors  
- Blocked pages → empty context  
- Context overflow → goal drift  

---

## 3. Technical Deep Dive

### Agent Loop (ReAct Pattern)

**What it does:**  
Drives the entire system by repeatedly calling the model and executing tool calls.

**Critical code:**
```python
for _ in range(MAX_ROUNDS):
        response = client.chat.completions.create(...)
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        for tc in msg.tool_calls:
            result = execute_tool(
                tc.function.name,
                json.loads(tc.function.arguments)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
```

**Why this works:**  
The model decides actions, but your loop enforces execution.

**Analogy:**  
Manager → worker → report → repeat.

---

### Tool: web_search (Tavily)

**What it does:**  
Returns top search results with clean snippets.

**Example:**  
Query: “What is RAG?” → returns results with summaries  

---

### Tool: fetch_page

**What it does:**  
Fetches full page content asynchronously.

**Critical code:**
    pages = await asyncio.gather(*tasks)

**Plain English:**  
Fetch multiple pages in parallel instead of waiting one by one.

**Failure case:**  
JS-heavy pages → empty content  

---

### Tool: calculate (AST Safe Eval)

**What it does:**  
Safely evaluates math expressions.

**Critical code:**
    if "__" in expression:
        raise ValueError("Unsafe expression")

**Why:**  
Prevents execution of malicious code.

---

### Tool: write_report

**What it does:**  
Writes structured markdown output to disk.

**Why:**  
Forces structured output instead of vague text.

---

### Streaming

**What it does:**  
Streams tokens in real time.

**Behavior:**
- Visible during reasoning  
- Silent during tool calls  

Plain English:  
Like watching the model think live.

---

## 4. Challenges, Failures, and Pivots

### Blocked Pages
- Problem: JS pages return empty content  
- Impact: degraded reports  
- Fix: deferred to Phase 2  

---

### Context Window Limits
- Problem: large content pushes out earlier context  
- Impact: model forgets original goal  
- Fix: solved in Phase 2 via retrieval  

---

### No Verification
- Problem: model trusts single source  
- Fix: solved in Phase 2  

---

### Tool Call Fragility
- Problem: strict API ordering requirements  
- Fix: careful message handling  

---

## 5. Evaluation & Results

| Metric | Result |
|------|--------|
| Tool execution reliability | High |
| Report quality | Moderate |
| Latency | ~40–70s (observed runs) |
| Failure modes | Blocked pages, context drift |

**Interpretation:**  
System is reliable but heavily dependent on input quality.

---

## 6. Key Learnings & Takeaways

- Principle: Agents are loops, not intelligence  
  Example: removing the loop breaks everything  

- Principle: Tool schemas define behavior  
  Example: better descriptions improve tool selection  

- Principle: Context is the bottleneck  
  Example: large pages cause goal drift  

- Principle: External data is unreliable  
  Example: model repeats incorrect sources  

- Principle: Safety must be explicit  
  Example: AST prevents code execution  

---

## 7. What’s Next (Phase 2)

Phase 1 limitations:
- No memory  
- No fallback  
- No verification  

Phase 2 introduces:
- Knowledge base  
- Retrieval (RAG)  
- Source grounding  

---

## 8. Appendix: How to Run
```
git clone https://github.com/anudeepreddy332/cli-research-agent.git  
uv sync  
uv run python main.py  
```


Code reference: main.py, src/research_agent/tools.py

## 9. Post‑Phase 1: Rebuilding with LangGraph (Learning Extension)

After completing Phase 2 (RAG + memory), I returned to Phase 1 and rebuilt the same agent using **LangGraph**. This was an intentional learning exercise to understand how agentic frameworks differ from raw loops.

### Why Rebuild the Same Agent?

The original `for` loop worked perfectly, but it had implicit control flow. Adding a new step (e.g., a verification tool) would require nested conditionals and scattered state flags. LangGraph makes the state machine **explicit**:

- Every piece of data is declared in a typed `AgentState`.
- Every step is a **node** (a pure function).
- Transitions are defined as **edges** (unconditional) or **conditional edges** (routing functions).

### LangGraph Architecture for the Research Agent
    START → call_model → should_continue?
    ├─ tool_executor → call_model (loop)
    └─ END


**Nodes:**
- `call_model` – Invokes DeepSeek with the current message history and available tools.
- `tool_executor` – Executes any tool calls requested by the model and appends results.

**Conditional Edge (`should_continue`):**
- Routes to `tool_executor` if the model requested a tool call.
- Routes to `END` if the model finished (or cost/iteration limits exceeded).

**State Fields (AgentState):**
- `messages` – Conversation history (list of dicts).
- `fetch_called` – Guard to ensure `fetch_page` runs at most once.
- `report_written` – Flag to terminate after `write_report`.
- `iterations`, `total_cost_usd`, `total_tokens` – Observability metrics.

### Key Differences Observed

| Aspect | Original Raw Loop | LangGraph Rebuild |
|--------|-------------------|-------------------|
| Token streaming | Real‑time visible output. | Not implemented (prints node transitions only). |
| Cost tracking | Not originally built. | Accumulated and displayed after run. |
| State inspection | Manual print statements. | Built‑in `get_state()` method. |
| Extensibility | Adding a step requires modifying loop logic. | Add a node and wire an edge. |

The LangGraph version produces **identical quality reports** but with better observability and a foundation for more complex agents (Phase 3+).

### Running the LangGraph Version

```bash
uv run python main_langgraph.py
```
The workflow is identical to the original. A test question like "What is the capital of France?" will produce a report in the same reports/ directory.

### Lessons from the Rebuild
- Frameworks add structure. LangGraph forces you to declare your state machine, which reduces bugs as complexity grows.

- Explicit state is easier to debug. You always know exactly what data is flowing.

- Raw loops are perfectly fine for simple agents. The right tool depends on the complexity of the task.