import json


SYSTEM_PROMPT = """
    You are a browser automation agent. You receive a task and the current 
    state of the browser (URL, page title, and visible elements with their indices).

    You must respond with ONLY valid JSON in this exact format:
    {
    "current_state": {
        "evaluation_previous_goal": "description of what happened",
        "memory": "relevant info to remember",
        "next_goal": "what to do next"
    },
    "action": [
        {"action_name": {"param": "value"}}
    ]
    }

    Available actions:
    - go_to_url: {"url": "https://..."} 
    - click_element: {"index": <element_index>}
    - input_text: {"index": <element_index>, "text": "value"}
    - scroll_down: {}
    - scroll_up: {}
    - go_back: {}
    - done: {"text": "extracted result"}
"""

def convert_example(task, steps, turn_index=0):
    conversations = []
    user_msg = f"""
        Task: {task}
        Current browser state:
        URL: about:blank
        Title: New Tab
        Available elements:
        [0] <input type="text" name="url" placeholder="Enter URL"/>
        [1] <button>Go</button>
    """

    step = steps[0]
    action_type = step.split(":")[0].strip()
    action_value = ":".join(step.split(":")[1:]).strip()

    if action_type == "go_to_url":
        assistant_response = json.dumps({
            "current_state": {
                "evaluation_previous_goal": "Starting task",
                "memory": f"Task: {task}",
                "next_goal": f"Navigate to {action_value}"
            },
            "action": [{"go_to_url": {"url": action_value}}]
        })
    elif action_type == "click_element":
        assistant_response = json.dumps({
            "current_state": {
                "evaluation_previous_goal": "Page Loaded",
                "memory": f"Task: {task}",
                "next_goal": f"Click on {action_value}"
            },
            "action": [{"click_element": {"index": 0}}]
        })
    elif action_type == "input_text":
        text_val = action_value.split(" in ")[0].strip()
        assistant_response = json.dumps({
            "current_state": {
                "evaluation_previous_goal": "Page loaded",
                "memory": f"Task: {task}",
                "next_goal": f"Enter text: {text_val}"
            },
            "action": [{"input_text": {"index": 0, "text": text_val}}]
        })
    elif action_type == "done":
        assistant_response = json.dumps({
            "current_state": {
                "evaluation_previous_goal": "Information found",
                "memory": f"Task: {task}",
                "next_goal": "Extract result and finish"
            },
            "action": [{"done": {"text": action_value}}]
        })

    conversations.append({
        "role": "user",
        "content": user_msg
    })
    conversations.append({
        "role": "assistant", 
        "content": assistant_response
    })

    return conversations


def build_training_data(json_file, output_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    training_data = []
    for example in data:
        for i, step in enumerate(example["steps"]):
            convos = convert_example(example["task"], example["steps"][i:], turn_index=i)
            training_data.append({
                "messages": [
                    {"role": "system",
                     "content": SYSTEM_PROMPT},
                     *convos
                ]
            })

    with open(output_file, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print("Generated training examples")


if __name__ == "__main__":
    build_training_data("few_shot_examples.json", "training_data.jsonl")