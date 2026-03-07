from flask import Flask, render_template, request, jsonify
from browser_use import Agent, Browser, SystemPrompt
from browser_use.browser.browser import BrowserConfig
from langchain_ollama import ChatOllama
import asyncio
import nest_asyncio
nest_asyncio.apply()
import json
import inspect


app = Flask(__name__)

SYSTEM_PROMPT = """You are a web research assistant. Follow these rules:

1. If the user provides a URL, go directly to that URL.
2. If no URL is provided, go to https://www.google.com and search for relevant keywords to find the right webpage.
3. Always proceed with your best judgment.
4. Extract the requested information thoroughly and accurately.
5. When you have collected all the information, present it clearly and say done.
"""

llm = ChatOllama(model="kimi-k2.5:cloud", temperature=0)

class CustomSystemPrompt(SystemPrompt):
    def important_rules(self):
        existing = super().important_rules()
        return existing + "\nSYSTEM_PROMPT"
    

def generate_subtasks(question):
    prompt = f"""
        Can you break the following task into a set of instructions so that a browser agent can easily follow the instructions to complete the task.
        Task: {question}
    """
    response = llm.invoke(prompt)
    return response.content


async def run_agent(task):
    browser = Browser(config=BrowserConfig(headless=True))
    prompt = generate_subtasks(task)
    print("Prompt: ", prompt)
    agent = Agent(
        llm=llm, 
        task=prompt, 
        browser=browser,
        system_prompt_class=CustomSystemPrompt,
        max_failures=10,
        max_actions_per_step=1,
    )

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
    task = "Go to https://www.cs.umd.edu/people/faculty and Fetch me the name and possibly the website of the research lab run by professor Zhicheng Liu"
    task, result, agent = asyncio.run(run_agent(message))

    prompt_object = {
        "prompt": task,
        "answer": str(result.final_result())
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