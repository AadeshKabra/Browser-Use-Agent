import json
from pathlib import Path


TRACE_DIR = Path("traces/")
TRACE_DIR.mkdir(exist_ok=True)


def save_trace(task, trace_steps):
    """
    Each trace_step should have:
    {
        "browser_state": { "url": "...", "title": "...", "elements": [...] },
        "llm_input": "the full prompt sent to LLM",
        "llm_output": "the raw JSON response from LLM",
        "action_taken": { "action_name": { "param": "value" } },
        "success": true/false
    }
    """
    trace = {"task": task, "steps": trace_steps}
    filename = TRACE_DIR / f"trace_{len(list(TRACE_DIR.glob('*.json'))):04d}.json"
    with open(filename, "w") as f:
        json.dump(trace, f, indent=2)

    print("Saved Trace")


def traces_to_training_data(trace_dir, output_file):
    traces = sorted(Path(trace_dir).glob("*.json"))

    training_data = []

    for trace_file in traces:
        with open(trace_file, "r") as f:
            trace = json.load(f)

        if not all (s.get("success", True) for s in trace["steps"]):
            continue

        for step in trace["steps"]:
            training_data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": step["llm_input"]},
                    {"role": "assistant", "content": step["llm_output"]}
                ]
            }) 


    with open(output_file, "w") as f:
        for ex in training_data:
            f.write(json.dumps(ex) + "\n")

    print("Generated examples from traces")