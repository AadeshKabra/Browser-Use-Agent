"""
Generate synthetic browser agent traces using an LLM.
No browser needed — just generates realistic trace data for fine-tuning.

Usage:
  python generate_synthetic_traces.py              # uses Ollama (kimi-k2.5)
  python generate_synthetic_traces.py --groq       # uses Groq (free, faster)
  python generate_synthetic_traces.py --start 10   # resume from task 10
"""

import json
import time
import sys
import re
from pathlib import Path

# ── Parse args ──────────────────────────────────────────────────
use_groq = "--groq" in sys.argv
start_from = 0
for i, arg in enumerate(sys.argv):
    if arg == "--start" and i + 1 < len(sys.argv):
        start_from = int(sys.argv[i + 1])

# ── Setup LLM ──────────────────────────────────────────────────
if use_groq:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=""  # replace with your key
    )
    delay_between = 20  # Groq free tier needs longer gaps
    print("Using Groq (llama-3.3-70b)")
else:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="kimi-k2.5:cloud", temperature=0.3)
    delay_between = 15  # longer delay to avoid rate limits
    print("Using Ollama (kimi-k2.5:cloud)")

# ── Prompt ──────────────────────────────────────────────────────
GENERATION_PROMPT = """You are generating training data for a browser automation model.

Given a task, generate a realistic multi-step browser agent trace. Each step represents 
what the agent sees and does at each point during web browsing.

For each step provide:
1. browser_state: what the browser currently shows
   - url: the current page URL  
   - title: the page title
2. llm_output: what the agent thinks and does
   - current_state: a string with these fields:
     page_summary='what the page shows'
     evaluation_previous_goal='Success/Failed - what happened'  
     memory='what to remember'
     next_goal='what to do next'
   - action: exactly ONE action as a string in this format:
     For navigation: "[ActionModel(go_to_url=GoToUrlAction(url='https://...'))]"
     For clicking: "[ActionModel(click_element=ClickElementAction(index=N, xpath=None))]"
     For typing: "[ActionModel(input_text=InputTextAction(index=N, text='search query'))]"
     For scrolling: "[ActionModel(scroll_down=ScrollAction(amount=None))]"
     For finishing: "[ActionModel(done=DoneAction(text='the final answer'))]"
     (All other fields in ActionModel should be None)
3. success: true

Rules:
- Use realistic URLs that actually exist
- Use realistic element indices (between 1-200)
- The first step usually starts at about:blank
- The last step must use DoneAction with the answer
- Keep traces between 3-8 steps
- Be specific in the done text — give a concrete answer, not "Extract info"

Respond with ONLY a JSON array of steps, no markdown, no explanation.

Task: {task}"""

# ── Load tasks ──────────────────────────────────────────────────
with open("trace_collection_tasks.json") as f:
    tasks = json.load(f)

output_dir = Path("traces")
output_dir.mkdir(exist_ok=True)

# Check how many traces already exist
existing = list(output_dir.glob("trace_*.json"))
print(f"Found {len(existing)} existing traces")

# Load already-completed tasks to skip them
completed_tasks = set()
for ef in existing:
    try:
        with open(ef) as f:
            completed_tasks.add(json.load(f)["task"])
    except:
        pass
print(f"Already completed: {len(completed_tasks)} tasks")

# ── Generate ────────────────────────────────────────────────────
success_count = 0
fail_count = 0

for i, task_obj in enumerate(tasks):
    if i < start_from:
        continue

    task = task_obj["task"]

    if task in completed_tasks:
        print(f"\n[{i+1}/{len(tasks)}] ⏭️ Already done: {task[:55]}")
        continue

    print(f"\n[{i+1}/{len(tasks)}] {task[:65]}")

    try:
        # Retry with backoff on rate limits
        response = None
        for attempt in range(5):
            try:
                response = llm.invoke(GENERATION_PROMPT.format(task=task))
                break
            except Exception as api_err:
                if "429" in str(api_err):
                    wait = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s, 150s
                    print(f"  ⏳ Rate limited, waiting {wait}s (attempt {attempt+1}/5)")
                    time.sleep(wait)
                else:
                    raise api_err

        if response is None:
            raise Exception("Failed after 5 retries")

        content = response.content.strip()

        # Strip markdown code fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        steps = json.loads(content)

        # Validate basic structure
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError("Empty or non-list response")

        # Check last step has DoneAction
        last_action = steps[-1].get("llm_output", {}).get("action", "")
        if "DoneAction" not in str(last_action):
            print(f"  ⚠️ Last step missing DoneAction, adding it")
            steps[-1]["llm_output"]["action"] = (
                "[ActionModel(done=DoneAction(text='Task completed - information extracted'))]"
            )

        # Ensure all steps have success field
        for step in steps:
            if "success" not in step:
                step["success"] = True

        trace = {"task": task, "steps": steps}

        # Save individual trace
        trace_num = len(list(output_dir.glob("*.json")))
        trace_file = output_dir / f"trace_{trace_num:04d}.json"
        with open(trace_file, "w") as f:
            json.dump(trace, f, indent=2)

        success_count += 1
        print(f"  ✅ {len(steps)} steps → {trace_file.name}")

    except json.JSONDecodeError as e:
        fail_count += 1
        print(f"  ❌ Invalid JSON: {str(e)[:80]}")
        # Save raw response for debugging
        debug_file = output_dir / f"failed_{i:04d}.txt"
        with open(debug_file, "w") as f:
            f.write(content)
        print(f"     Raw response saved to {debug_file}")

    except Exception as e:
        fail_count += 1
        print(f"  ❌ Error: {str(e)[:80]}")

    # Rate limit delay
    time.sleep(delay_between)

# ── Summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"DONE: {success_count} succeeded, {fail_count} failed out of {len(tasks)} tasks")
print(f"Traces saved in: {output_dir}/")
print(f"{'='*60}")

# ── Combine all traces into one file ────────────────────────────
all_traces = []
for trace_file in sorted(output_dir.glob("trace_*.json")):
    with open(trace_file) as f:
        all_traces.append(json.load(f))

with open("all_synthetic_traces.json", "w") as f:
    json.dump(all_traces, f, indent=2)

print(f"Combined {len(all_traces)} traces → all_synthetic_traces.json")