"""
LangGraph version of CLI Research Agent
Usage: python -m main_langgraph
"""
import time

from src.research_agent.graph import build_graph, SYSTEM_PROMPT


def main():
    question = input("Enter your research question: ").strip()
    if not question:
        print("No question provided.")
        return

    graph = build_graph()

    initial_state = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "question": question,
        "fetch_called": False,
        "report_written": False,
        "iterations": 0,
        "total_cost_usd": 0.0,
        "total_tokens": 0,
        "status": "running",
    }

    print(f"\n=== LangGraph Research Agent ===\n")
    start = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start

    print("\n=== Run Summary ===")
    print(f"Status:        {final_state.get('status', 'unknown')}")
    print(f"Iterations:    {final_state['iterations']}")
    print(f"Total tokens:  {final_state['total_tokens']}")
    print(f"Total cost:    ${final_state['total_cost_usd']:.4f}")
    print(f"Elapsed:       {elapsed:.1f}s")

    if final_state.get("status") == "done":
        print("\n✓ Report generated. Check the 'reports/' directory.")

if __name__ == "__main__":
        main()