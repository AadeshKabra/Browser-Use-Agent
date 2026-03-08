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


app = Flask(__name__)

SYSTEM_PROMPT = """You are a web research assistant. Follow these rules:

1. If the user provides a URL, go directly to that URL.
2. If no URL is provided, go to https://www.google.com and search for relevant keywords to find the right webpage.
3. Always proceed with your best judgment.
4. Extract the requested information thoroughly and accurately.
5. When you have collected all the information, present it clearly and say done.
"""

llm = ChatOllama(model="kimi-k2.5:cloud", temperature=0)

live_memory = deque(maxlen=100)
memory_lock = Lock()

class CustomSystemPrompt(SystemPrompt):
    def important_rules(self):
        existing = super().important_rules()
        return existing + "\nSYSTEM_PROMPT"


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
    # prompt = f"""
    #     Can you break the following task into a set of instructions so that a browser agent can easily follow the instructions to complete the task.
    #     Task: {question}
    # """

    prompt = f"""You are a browser agent. Complete tasks efficiently in as few steps as possible.
    Go directly to relevant URLs when possible instead of searching Google.

    {few_shot_string}

    Now complete this task efficiently:
    Task: {question}
    """
    response = llm.invoke(prompt)
    return response.content


async def run_agent(task):
    browser = Browser(config=BrowserConfig(headless=True))
    prompt = generate_subtasks(task)
    # print("Prompt: ", prompt)
    agent = Agent(
        llm=llm, 
        task=prompt, 
        browser=browser,
        system_prompt_class=CustomSystemPrompt,
        max_failures=10,
        max_actions_per_step=1,
    )

    # @agent.register_new_step_callback
    # async def on_step(step):
    #     if step.model_output:
    #         live_memory.append({
    #             "step": step.step_number if hasattr(step, 'step_number') else len(live_memory),
    #             "memory": str(step.model_output.current_state) if hasattr(step.model_output, 'current_state') else str(step.model_output),
    #             "action": str(step.model_output.action) if hasattr(step.model_output, 'action') else None,
    #         })

    # print([m for m in dir(agent) if not m.startswith('_')])

    def on_step(state, model_output, step_info=None):
        print(f">>> CALLBACK FIRED")  # Debug: confirm it's being called
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
    return task, result, agent



@app.route("/")
def main():
    # print(inspect.signature(Browser.__init__))
    return render_template("index.html")



@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get('message')
    # task = "Go to https://www.cs.umd.edu/people/faculty and Fetch me the name and possibly the website of the research lab run by professor Zhicheng Liu"
    task, result, agent = asyncio.run(run_agent(message))

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

