from flask import Flask, render_template, request, jsonify
from browser_use import Agent, Browser
from langchain_ollama import ChatOllama
import asyncio
import nest_asyncio
nest_asyncio.apply()
import json


app = Flask(__name__)


async def run_agent(task):
    llm = ChatOllama(model="kimi-k2.5:cloud", temperature=0)
    
    # browser = Browser(BrowserProfile(
    #     headless = True,
    #     disable_security = True
    # ))

    # task = "Fetch me the name and possibly the website of the research lab run by professor Zhicheng Liu"
    agent = Agent(
        llm=llm, 
        task=task, 
        # browser=browser,
        max_failures=10,
        max_actions_per_step=1,
    )

    result = await agent.run()
    # await browser.close()
    return task, result, agent



@app.route("/")
def main():
    return render_template("index.html")



@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get('message')
    task = "Go to https://www.cs.umd.edu/people/faculty and Fetch me the name and possibly the website of the research lab run by professor Zhicheng Liu"
    task, result, agent = asyncio.run(run_agent(message))
    print("======================================")
    print(result)
    print("======================================")
    final_answer = "No result found."
    for output in reversed(list(result.model_outputs())):
        if isinstance(output, dict) and 'done' in output and output['done'].get('text'):
            final_answer = output['done']['text']
            break

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