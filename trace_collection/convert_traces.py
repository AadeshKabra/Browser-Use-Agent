# """
# Convert browser-use trace JSON files into JSONL training data
# for fine-tuning a chat model (e.g., Qwen3-8B via Unsloth/LoRA).

# Usage:
#     python convert_traces.py --traces_dir ./traces --output training_data.jsonl
# """

# import json
# import glob
# import argparse
# from pathlib import Path


# SYSTEM_PROMPT = (
#     "You are a browser automation assistant. You help users accomplish tasks by "
#     "controlling a web browser. At each step, you receive the current browser state "
#     "(URL, page title, and visible elements). You must analyze the situation, "
#     "remember your progress, and decide the next action to take.\n\n"
#     "Respond with your current analysis and the action to execute. Your analysis "
#     "should include:\n"
#     "- page_summary: Brief description of what you see\n"
#     "- evaluation_previous_goal: Did the last action succeed?\n"
#     "- memory: Key facts to remember for the task\n"
#     "- next_goal: What to do next\n\n"
#     "Then provide the action as a JSON action model."
# )


# def format_browser_state(step, step_num):
#     """Format browser state into a user message."""
#     bs = step["browser_state"]
#     lines = [
#         f"[Step {step_num}] Current browser state:",
#         f"  URL: {bs['url']}",
#         f"  Title: {bs.get('title', '(none)')}",
#         f"  Page elements: {bs.get('elements', '(none)')}",
#     ]
#     return "\n".join(lines)


# # def format_llm_output(step):
# #     """Format LLM output into an assistant message."""
# #     llm = step["llm_output"]
# #     lines = [
# #         llm["current_state"],
# #         "",
# #         f"Action: {llm['action']}",
# #     ]
# #     return "\n".join(lines)

# def parse_action_model(action_str):
#     """Extract the non-None action from the ActionModel string."""
#     # Map of action patterns to extract
#     import re
    
#     action_types = {
#         "go_to_url": r"GoToUrlAction\(url='([^']+)'\)",
#         "click_element": r"ClickElementAction\(index=(\d+)",
#         "input_text": r"InputTextAction\(index=(\d+), text='([^']+)'",
#         "extract_content": r"extract_content_parameters\(goal='([^']+)'\)",
#         "done": r"DoneAction\(text='(.*?)'\)",
#         "scroll_down": r"scroll_down=ScrollAction",
#         "scroll_up": r"scroll_up=ScrollAction",
#         "search_google": r"SearchGoogleAction\(query='([^']+)'\)",
#     }
    
#     for action_type, pattern in action_types.items():
#         match = re.search(pattern, action_str, re.DOTALL)
#         if match:
#             if action_type == "go_to_url":
#                 return {"type": "go_to_url", "url": match.group(1)}
#             elif action_type == "click_element":
#                 return {"type": "click_element", "index": int(match.group(1))}
#             elif action_type == "input_text":
#                 return {"type": "input_text", "index": int(match.group(1)), "text": match.group(2)}
#             elif action_type == "extract_content":
#                 return {"type": "extract_content", "goal": match.group(1)}
#             elif action_type == "done":
#                 return {"type": "done", "text": match.group(1)}
#             elif action_type == "search_google":
#                 return {"type": "search_google", "query": match.group(1)}
    
#     return {"type": "unknown", "raw": action_str}


# def format_llm_output(step):
#     """Format LLM output as structured JSON for training."""
#     llm = step["llm_output"]
    
#     # Parse the action string to extract the actual action
#     action_str = llm["action"]
#     action = parse_action_model(action_str)
    
#     output = {
#         "current_state": llm["current_state"],
#         "action": action
#     }
#     return json.dumps(output, indent=2)


# def trace_to_messages(trace):
#     """Convert a single trace dict into a messages list for chat fine-tuning."""
#     messages = []

#     # System prompt
#     messages.append({"role": "system", "content": SYSTEM_PROMPT})

#     task = trace["task"]
#     steps = trace["steps"]

#     for i, step in enumerate(steps):
#         # Build user message: task context (first step) + browser state
#         if i == 0:
#             user_content = f"Task: {task}\n\n{format_browser_state(step, i)}"
#         else:
#             user_content = format_browser_state(step, i)

#         messages.append({"role": "user", "content": user_content})

#         # Assistant response: reasoning + action
#         assistant_content = format_llm_output(step)
#         messages.append({"role": "assistant", "content": assistant_content})

#     return messages


# def convert_traces(traces_dir, output_path, skip_failed=True):
#     """Convert all trace JSON files in a directory to a JSONL training file."""
#     trace_files = sorted(glob.glob(str(Path(traces_dir) / "trace_*.json")))

#     if not trace_files:
#         print(f"No trace files found in {traces_dir}")
#         return

#     # Check for failed traces file
#     failed_ids = set()
#     failed_file = Path(traces_dir) / "failed_0032.txt"
#     if failed_file.exists() and skip_failed:
#         with open(failed_file) as f:
#             for line in f:
#                 line = line.strip()
#                 if line:
#                     failed_ids.add(line)
#         print(f"Found {len(failed_ids)} failed trace IDs to skip")

#     converted = 0
#     skipped = 0
#     total_turns = 0

#     with open(output_path, "w") as out:
#         for fpath in trace_files:
#             fname = Path(fpath).stem  # e.g., "trace_0016"

#             # Skip failed traces if requested
#             if skip_failed and fname in failed_ids:
#                 skipped += 1
#                 continue

#             try:
#                 with open(fpath) as f:
#                     trace = json.load(f)
#             except (json.JSONDecodeError, KeyError) as e:
#                 print(f"  Skipping {fname}: {e}")
#                 skipped += 1
#                 continue

#             # Skip traces where any step failed (optional strictness)
#             all_success = all(s.get("success", False) for s in trace.get("steps", []))
#             if not all_success:
#                 print(f"  Skipping {fname}: contains failed steps")
#                 skipped += 1
#                 continue

#             messages = trace_to_messages(trace)
#             num_turns = len([m for m in messages if m["role"] == "assistant"])

#             out.write(json.dumps({"messages": messages}) + "\n")
#             converted += 1
#             total_turns += num_turns

#     print(f"\nDone!")
#     print(f"  Converted: {converted} traces")
#     print(f"  Skipped:   {skipped} traces")
#     print(f"  Total assistant turns: {total_turns}")
#     print(f"  Avg turns per trace:   {total_turns / max(converted, 1):.1f}")
#     print(f"  Output: {output_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--traces_dir", default="./traces", help="Directory with trace_*.json files")
#     parser.add_argument("--output", default="training_data.jsonl", help="Output JSONL path")
#     parser.add_argument("--include_failed_steps", action="store_true",
#                         help="Include traces that have failed steps")
#     args = parser.parse_args()

#     convert_traces(args.traces_dir, args.output, skip_failed=not args.include_failed_steps)








"""
Convert browser-use trace JSON files into JSONL training data
that matches browser-use's AgentOutput schema exactly.

Output format per assistant turn:
{
  "current_state": {
    "page_summary": "...",
    "evaluation_previous_goal": "...",
    "memory": "...",
    "next_goal": "..."
  },
  "action": [{"go_to_url": {"url": "https://..."}}]
}

Usage:
    python convert_traces_v2.py --traces_dir ./traces --output training_data.jsonl
"""

import json
import glob
import re
import argparse
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a browser automation assistant. You help users accomplish tasks by "
    "controlling a web browser. At each step, you receive the current browser state "
    "(URL, page title, and visible elements). You must analyze the situation, "
    "remember your progress, and decide the next action to take.\n\n"
    "Respond with a JSON object containing:\n"
    "- current_state: with page_summary, evaluation_previous_goal, memory, next_goal\n"
    "- action: array with one action object\n\n"
    "Available actions: go_to_url, click_element, input_text, extract_content, "
    "search_google, scroll_down, scroll_up, go_back, send_keys, done, switch_tab, "
    "open_tab, scroll_to_text, get_dropdown_options, select_dropdown_option"
)


def parse_current_state(state_str):
    """
    Parse the current_state string into AgentBrain JSON.
    Input:  "page_summary='...' evaluation_previous_goal='...' memory='...' next_goal='...'"
    Output: {"page_summary": "...", "evaluation_previous_goal": "...", "memory": "...", "next_goal": "..."}
    """
    fields = {}
    # Match field_name='value' patterns, handling nested quotes
    # We look for known field names and extract everything between them
    field_names = ["page_summary", "evaluation_previous_goal", "memory", "next_goal"]
    
    for i, field in enumerate(field_names):
        # Find the start of this field
        pattern = field + r"='?"
        match = re.search(pattern, state_str)
        if not match:
            fields[field] = ""
            continue
        
        start = match.end()
        
        # Find the start of the next field, or end of string
        if i + 1 < len(field_names):
            next_pattern = r"'\s+" + field_names[i + 1] + r"="
            next_match = re.search(next_pattern, state_str[start:])
            if next_match:
                value = state_str[start:start + next_match.start()]
            else:
                value = state_str[start:]
        else:
            value = state_str[start:]
        
        # Clean up trailing quote
        value = value.strip()
        if value.endswith("'"):
            value = value[:-1]
        
        fields[field] = value
    
    return fields


def parse_action(action_str):
    """
    Parse the ActionModel string into browser-use's action array format.
    Input:  "[ActionModel(done=None, go_to_url=GoToUrlAction(url='https://...'), ...)]"
    Output: [{"go_to_url": {"url": "https://..."}}]
    """
    # Map of action patterns and their parameter extractors
    action_patterns = [
        (
            "go_to_url",
            r"GoToUrlAction\(url='([^']+)'\)",
            lambda m: {"url": m.group(1)}
        ),
        (
            "click_element",
            r"ClickElementAction\(index=(\d+)",
            lambda m: {"index": int(m.group(1))}
        ),
        (
            "input_text",
            r"InputTextAction\(index=(\d+),\s*text='([^']*)'",
            lambda m: {"index": int(m.group(1)), "text": m.group(2)}
        ),
        (
            "extract_content",
            r"extract_content_parameters\(goal='([^']+)'\)",
            lambda m: {"goal": m.group(1)}
        ),
        (
            "done",
            r"DoneAction\(text='(.*?)(?:'\))",
            lambda m: {"text": m.group(1)}
        ),
        (
            "search_google",
            r"SearchGoogleAction\(query='([^']+)'\)",
            lambda m: {"query": m.group(1)}
        ),
        (
            "scroll_down",
            r"scroll_down=ScrollAction\(amount=(\d+)\)",
            lambda m: {"amount": int(m.group(1))}
        ),
        (
            "scroll_up",
            r"scroll_up=ScrollAction\(amount=(\d+)\)",
            lambda m: {"amount": int(m.group(1))}
        ),
        (
            "go_back",
            r"go_back=GoBackAction\(\)",
            lambda m: {}
        ),
        (
            "send_keys",
            r"SendKeysAction\(keys='([^']+)'\)",
            lambda m: {"keys": m.group(1)}
        ),
        (
            "switch_tab",
            r"SwitchTabAction\(page_id=(\d+)\)",
            lambda m: {"page_id": int(m.group(1))}
        ),
        (
            "open_tab",
            r"OpenTabAction\(url='([^']+)'\)",
            lambda m: {"url": m.group(1)}
        ),
        (
            "scroll_to_text",
            r"ScrollToTextAction\(text='([^']+)'\)",
            lambda m: {"text": m.group(1)}
        ),
    ]
    
    for action_name, pattern, extractor in action_patterns:
        match = re.search(pattern, action_str, re.DOTALL)
        if match:
            params = extractor(match)
            return [{action_name: params}]
    
    # Fallback: try to find any non-None action
    # Look for pattern: action_name=SomeAction(...)
    fallback = re.search(r'(\w+)=(\w+Action)\(([^)]*)\)', action_str)
    if fallback:
        action_name = fallback.group(1)
        return [{action_name: {"raw": fallback.group(0)}}]
    
    return [{"unknown": {"raw": action_str[:200]}}]


def format_browser_state(step, step_num):
    """Format browser state into a user message."""
    bs = step["browser_state"]
    lines = [
        f"[Step {step_num}] Current browser state:",
        f"  URL: {bs['url']}",
        f"  Title: {bs.get('title', '(none)')}",
        f"  Page elements: {bs.get('elements', '(none)')}",
    ]
    return "\n".join(lines)


def format_llm_output(step):
    """
    Format LLM output as AgentOutput JSON matching browser-use's schema.
    """
    llm = step["llm_output"]
    
    current_state = parse_current_state(llm["current_state"])
    action = parse_action(llm["action"])
    
    output = {
        "current_state": current_state,
        "action": action
    }
    
    return json.dumps(output)


def trace_to_messages(trace):
    """Convert a single trace dict into a messages list for chat fine-tuning."""
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})

    task = trace["task"]
    steps = trace["steps"]

    for i, step in enumerate(steps):
        if i == 0:
            user_content = f"Task: {task}\n\n{format_browser_state(step, i)}"
        else:
            user_content = format_browser_state(step, i)

        messages.append({"role": "user", "content": user_content})

        assistant_content = format_llm_output(step)
        messages.append({"role": "assistant", "content": assistant_content})

    return messages


def convert_traces(traces_dir, output_path, skip_failed=True):
    """Convert all trace JSON files in a directory to a JSONL training file."""
    trace_files = sorted(glob.glob(str(Path(traces_dir) / "trace_*.json")))

    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        return

    # Check for failed traces file
    failed_ids = set()
    failed_files = list(Path(traces_dir).glob("failed_*.txt"))
    if failed_files and skip_failed:
        for ff in failed_files:
            with open(ff) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        failed_ids.add(line)
        print(f"Found {len(failed_ids)} failed trace IDs to skip")

    converted = 0
    skipped = 0
    total_turns = 0
    parse_warnings = []

    with open(output_path, "w") as out:
        for fpath in trace_files:
            fname = Path(fpath).stem

            if skip_failed and fname in failed_ids:
                skipped += 1
                continue

            try:
                with open(fpath) as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Skipping {fname}: {e}")
                skipped += 1
                continue

            # Skip traces with failed steps
            all_success = all(s.get("success", False) for s in trace.get("steps", []))
            if not all_success:
                print(f"  Skipping {fname}: contains failed steps")
                skipped += 1
                continue

            messages = trace_to_messages(trace)
            
            # Validate: check for unknown actions
            for m in messages:
                if m["role"] == "assistant":
                    try:
                        parsed = json.loads(m["content"])
                        for act in parsed.get("action", []):
                            if "unknown" in act:
                                parse_warnings.append(f"{fname}: unknown action found")
                    except json.JSONDecodeError:
                        parse_warnings.append(f"{fname}: invalid JSON in output")

            num_turns = len([m for m in messages if m["role"] == "assistant"])
            out.write(json.dumps({"messages": messages}) + "\n")
            converted += 1
            total_turns += num_turns

    print(f"\nDone!")
    print(f"  Converted: {converted} traces")
    print(f"  Skipped:   {skipped} traces")
    print(f"  Total assistant turns: {total_turns}")
    print(f"  Avg turns per trace:   {total_turns / max(converted, 1):.1f}")
    print(f"  Output: {output_path}")
    
    if parse_warnings:
        print(f"\n  ⚠️  Warnings ({len(parse_warnings)}):")
        for w in parse_warnings:
            print(f"    - {w}")

    # Print a sample for verification
    print(f"\n--- Sample output (first assistant turn) ---")
    with open(output_path) as f:
        first = json.loads(f.readline())
        for m in first["messages"]:
            if m["role"] == "assistant":
                print(json.dumps(json.loads(m["content"]), indent=2))
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces_dir", default="../traces")
    parser.add_argument("--output", default="training_data.jsonl")
    parser.add_argument("--include_failed_steps", action="store_true")
    args = parser.parse_args()

    convert_traces(args.traces_dir, args.output, skip_failed=not args.include_failed_steps)