import datetime as dt
import json
import pathlib
import subprocess

QUESTIONS = [
    "What are the latest breakthroughs in LLM efficiency?",
    "What is the current state of quantum computing in 2025?",
    "What are the top open source AI coding assistants right now?",
    "How is AI being used in drug discovery?",
    "What are the best practices for RAG systems in production?",
    "What happened with AI regulation in the EU in 2024?",
    "What are the most energy-efficient large language models?",
    "How does speculative decoding work and who uses it?",
    "What are the latest robotics breakthroughs using AI?",
    "What is the state of multimodal AI models in 2025?",
]

results = []

for q in QUESTIONS:
    print(f"\nRunning: {q[:60]}...")
    start = dt.datetime.now()
    proc = subprocess.run(
        ["python", str(pathlib.Path(__file__).parent.parent / "main.py")],
        input=q,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(pathlib.Path(__file__).parent.parent)
    )
    elapsed = (dt.datetime.now() - start).total_seconds()
    report_files = sorted((pathlib.Path(__file__).parent.parent / "reports").glob("*.md"))
    last_report = report_files[-1].read_text() if report_files else ""
    results.append({
        "question": q,
        "elapsed_s": round(elapsed, 1),
        "report_chars": len(last_report),
        "stdout": proc.stdout[-500:],
        "error": proc.stderr[-300:] if proc.returncode != 0 else "",
    })
    print(f"  → {elapsed:.1f}s, {len(last_report)} chars")

out = pathlib.Path("benchmark_results.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nResults saved to {out}")
print(f"Avg chars: {sum(r['report_chars'] for r in results) // len(results)}")
print(f"Failures: {sum(1 for r in results if r['error'])}")
