from flask import Flask, render_template, request, jsonify
from browser_use import Agent, Browser, SystemPrompt
from browser_use.agent.service import Agent
from browser_use.agent.views import AgentOutput
from browser_use.browser.browser import BrowserConfig
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
import asyncio
import nest_asyncio
nest_asyncio.apply()
import json
import re         
import logging     
import inspect
import time
from collections import deque
from threading import Lock
from few_shot import FEW_SHOT_EXAMPLES
from trace_collection.collect_traces import save_trace
import random
import warnings



app = Flask(__name__)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="unclosed transport")
SYSTEM_PROMPT = """You are a web research assistant. Follow these rules:

1. If the user provides a URL, go directly to that URL.
2. If no URL is provided, go to https://www.google.com and search for relevant keywords to find the right webpage.
3. Always proceed with your best judgment.
4. Extract the requested information thoroughly and accurately.
5. When you have collected all the information, present it clearly and say done.
"""


llm = ChatOllama(model="kimi-k2.5:cloud", temperature=0.4)

live_memory = deque(maxlen=100)
memory_lock = Lock()


class CustomSystemPrompt(SystemPrompt):
    def __init__(self, *args, task_examples="", **kwargs):
        super().__init__(*args, **kwargs)
        self.task_examples = task_examples


    def important_rules(self):
        existing = super().important_rules()
        return existing + "\n" + SYSTEM_PROMPT + "\n" + self.task_examples


def classify_task(query):
    query = query.lower()
    keywords = {
        "extract_info": ["email", "phone", "address", "hours", "price", "cost" , "find the", "get the", "fetch", "what is the", "how much"],
        "search": ["search for", "look up", "find on", "jobs", "restaurants",
                   "trending", "top rated", "best", "most popular"],
        "form_fill": ["fill", "submit", "sign up", "register", "contact form",
                      "apply", "enter", "type in"],
        "navigation": ["latest", "newest", "blog", "news", "deadline",
                        "announcement", "check", "browse", "go to"],
        "comparison": ["compare", "vs", "versus", "difference between",
                        "which is better", "which has more"],
        "error_recovery": [],
    }

    scores = {}
    for category, words in keywords.items():
        scores[category] = sum(1 for w in words if w in query)

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    else:
        return "navigation"
    

def select_examples(query: str, k: int = 3, include_recovery: bool = False) -> list:
    category = classify_task(query)
    pool = list(FEW_SHOT_EXAMPLES.get(category, []))
 
    if len(pool) < k:
        fallback = list(FEW_SHOT_EXAMPLES.get("navigation", []))
        random.shuffle(fallback)
        pool.extend(fallback[:k - len(pool)])
 
    selected = random.sample(pool, min(k, len(pool)))
 
    if include_recovery and FEW_SHOT_EXAMPLES.get("error_recovery"):
        recovery = random.choice(FEW_SHOT_EXAMPLES["error_recovery"])
        if len(selected) >= k:
            selected[-1] = recovery 
        else:
            selected.append(recovery)
 
    return selected


def format_few_shot_examples(examples: list) -> str:
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}: {ex['task']}")
        for step in ex["steps"]:
            action = step["action"]
            if action == "go_to_url":
                lines.append(f"  → go_to_url: {step['url']}")
            elif action == "click_element":
                lines.append(f"  → click_element: {step['target']}")
            elif action == "input_text":
                lines.append(f"  → input_text: [{step['target']}] \"{step['text']}\"")
            elif action == "scroll_down":
                lines.append(f"  → scroll_down: {step['amount']} times")
            elif action == "select_dropdown":
                lines.append(f"  → select_dropdown: [{step['target']}] = \"{step['value']}\"")
            elif action == "extract_page_content":
                lines.append(f"  → extract_page_content")
            elif action == "go_back":
                lines.append(f"  → go_back")
            elif action == "done":
                lines.append(f"  → done: {step['text']}")
        lines.append("")
    return "\n".join(lines)


def memory_callback(data):
    with memory_lock:
        live_memory.append(data)


def generate_subtasks(question):
    # few_shot_string = format_few_shot_examples(FEW_SHOT_EXAMPLES)

    selected_few_shots = select_examples(question, 3, True)
    # print(selected_few_shots)
    few_shot_string = format_few_shot_examples(selected_few_shots)
    # print(few_shot_string)
    prompt = f"""You are a browser agent. Complete tasks efficiently in as few steps as possible.
    Go directly to relevant URLs when possible instead of searching Google.

    {few_shot_string}

    Now complete this task efficiently:
    Task: {question}
    """
    response = llm.invoke(prompt)
    # print("Response: ", response.content)
    return response.content


async def run_agent(task):
    with memory_lock:
        live_memory.clear()

    browser = Browser(config=BrowserConfig(headless=True))
    # prompt = generate_subtasks(task)
    # print("Prompt: ", prompt)
    selected_few_shots = select_examples(task, 3, True)
    # print(selected_few_shots)
    few_shot_string = format_few_shot_examples(selected_few_shots)

    live_traces = []
    print("Just before calling the agent")
    agent = Agent(
        llm=llm, 
        task=task,
        browser=browser,
        system_prompt_class=lambda *a, **kw: CustomSystemPrompt(*a, task_examples=few_shot_string, **kw),
        use_vision=False,
        max_input_tokens=32000,
        max_failures=10,
        max_actions_per_step=1,
    )

    def on_step(state, model_output, step_info=None):
        print(f">>> MEMORY UPDATED")  
        if model_output:
            with memory_lock:
                live_memory.append({
                    "step": len(live_memory) + 1,
                    "memory": str(model_output.current_state) if hasattr(model_output, 'current_state') else str(model_output),
                    "action": str(model_output.action) if hasattr(model_output, 'action') else None,
                })


    agent.register_new_step_callback = on_step


    result = await agent.run()
    # await browser.close()
    return task, result, agent, live_traces



@app.route("/")
def main():
    return render_template("index.html")



@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get('message')
    # task = "Go to https://www.cs.umd.edu/people/faculty and Fetch me the name and possibly the website of the research lab run by professor Zhicheng Liu"
    task, result, agent, trace_steps = asyncio.run(run_agent(message))

    if result.final_result() and trace_steps:
        try:
            save_trace(message, trace_steps)
        except Exception as e:
            print(f"Failed to save trace: {e}")

    memory_log = []
    for i, step in enumerate(result.history):
        if step.model_output:
            memory_log.append({
                "step": i,
                "memory": str(step.model_output.current_state) if step.model_output.current_state else None,
                "action": str(step.model_output.action) if step.model_output.action else None,
            })

    prompt_object = {
        "prompt": task,
        "answer": str(result.final_result()),
    }

    try:
        with open("tests.json", "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(prompt_object)
    with open("tests.json", "w") as f:
        json.dump(data, f, indent=2)

    return jsonify({"response": str(result.final_result())})


@app.route("/memory", methods=["GET"])
def get_memory():
    with memory_lock:
        return jsonify({"memory": list(live_memory)})
    

@app.route("/memory/stream")
def stream_memory():
    def generate():
        last_index = 0
        while True:
            with memory_lock:
                if len(live_memory) > last_index:
                    for item in list(live_memory)[last_index:]: 
                        yield f"data: {json.dumps(item)}\n\n"
                    last_index = len(live_memory)

            time.sleep(0.5)
    
    return app.response_class(generate(), mimetype='text/event-stream') 


if __name__ == "__main__":
    app.run(debug=True, threaded=True)

