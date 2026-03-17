from flask import Flask, render_template, request, jsonify
from browser_use import Agent, Browser, SystemPrompt
from browser_use.browser.browser import BrowserConfig
from langchain_ollama import ChatOllama
import asyncio
import nest_asyncio
nest_asyncio.apply()
import json
import inspect
import time
from collections import deque
from threading import Lock
from few_shot import FEW_SHOT_EXAMPLES
from trace_collection.collect_traces import save_trace
import sys
from langchain_groq import ChatGroq



SYSTEM_PROMPT = """You are a web research assistant. Follow these rules:

1. If the user provides a URL, go directly to that URL.
2. If no URL is provided, navigate directly to the most relevant website instead of searching Google.
   - For UMD info → go to umd.edu or cs.umd.edu directly
   - For tech docs → go to the official docs site directly
   - For prices → go to the store website directly
   - For academic papers → go to scholar.google.com or arxiv.org directly
3. NEVER go to google.com/search — it will be blocked by reCAPTCHA.
4. Extract the requested information thoroughly and accurately.
5. When you have collected all the information, present it clearly and say done.
"""

llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.4)

live_memory = deque(maxlen=100)
memory_lock = Lock()

trace_steps = []


class CustomSystemPrompt(SystemPrompt):
    def important_rules(self):
        existing = super().important_rules()
        return existing + "\n" + SYSTEM_PROMPT


def fortmat_few_shot_examples(examples, n=3):
    formatted = ""
    for ex in examples[:n]:
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(ex['steps']))
        formatted += f"\nExample:\nTask: {ex['task']}\nSteps:\n{steps}\n"

    return formatted


def memory_callback(data):
    with memory_lock:
        live_memory.append(data)


def generate_subtasks(question):
    few_shot_string = fortmat_few_shot_examples(FEW_SHOT_EXAMPLES)

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
    browser = Browser(config=BrowserConfig(headless=True))
    # prompt = generate_subtasks(task)
    # print("Prompt: ", prompt)

    live_traces = []

    agent = Agent(
        llm=llm, 
        task=task,
        browser=browser,
        system_prompt_class=CustomSystemPrompt,
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

                elements_str = None
                if state:
                    for attr in ['element_tree', 'elements', 'dom_tree', 'content']:
                        if hasattr(state, attr) and getattr(state, attr) is not None:
                            elements_str = str(getattr(state, attr))[:10000]  # truncate large DOMs
                            break

                live_traces.append({
                    "browser_state": {
                        "url": state.url if hasattr(state, 'url') else None,
                        "title": state.title if hasattr(state, 'title') else None,
                        "elements": elements_str,
                    },
                    "llm_output": {
                        "current_state": str(model_output.current_state) if hasattr(model_output, 'current_state') else None,
                        "action": str(model_output.action) if hasattr(model_output, 'action') else None,
                    },
                    "success": True,
                })

    agent.register_new_step_callback = on_step


    try:
        result = await asyncio.wait_for(agent.run(), timeout=180)  # 3 minutes
    except asyncio.TimeoutError:
        print(f"Task timed out after 3 minutes: {task[:60]}")
        result = None
    await browser.close()
    return task, result, agent, live_traces


trace_tasks = []

batch_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0
batch_size = 5

with open("trace_collection_tasks.json", "r") as f:
    trace_tasks = json.load(f)


start = batch_num * batch_size
end = min(start + batch_size, len(trace_tasks))
batch = trace_tasks[start:end]

for trace_task in batch:
    try:
        task, result, agent, trace_steps = asyncio.run(run_agent(trace_task['task']))

        if result and result.final_result() and trace_steps:
            try:
                save_trace(trace_task['task'], trace_steps)
            except Exception as e:
                print(f"Failed to save trace: {e}")

            # Only process history if we have a valid result
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

        else:
            print(f" No result for: {trace_task['task'][:60]}")

    except Exception as e:
        print(f"Error running task '{trace_task['task']}': {e}")
        continue