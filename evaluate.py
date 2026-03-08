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


llm = ChatOllama(model="kimi-k2.5:cloud", temperature=0)

live_memory = deque(maxlen=100)
memory_lock = Lock()

with open("few_shot_examples.json", "r") as f:
    few_shot = json.load(f)

with open("eval_tasks.json", "r") as f:
    eval_tasks = json.load(f)


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


async def run_with_few_shots(question):
    browser = Browser(config=BrowserConfig(headless=True))
    few_shot_string = fortmat_few_shot_examples(few_shot)
    prompt = f"""Complete this task efficiently in as few steps as possible.
    Go directly to relevant URLs when possible instead of searching Google.

    {few_shot_string}

    Task: {question}"""

    agent = Agent(llm=llm, task=prompt, browser=browser, max_failures=10, max_actions_per_step=1)
    result = await agent.run()
    await browser.close()
    return result


async def run_without_few_shots(question):
    browser = Browser(config=BrowserConfig(headless=True))
    prompt = f"""Complete this task efficiently in as few steps as possible.
    Go directly to relevant URLs when possible instead of searching Google.
    Task: {question}"""

    response = llm.invoke(prompt)
    agent = Agent(llm=llm, task=prompt, browser=browser, max_failures=10, max_actions_per_step=1)
    result = await agent.run()
    await browser.close()
    return result


def check_answer(result, expected):
    if expected is None:
        return result is not None
    if result is None:
        return False

    return expected.lower() in result.lower()


async def evaluate():
    results = []

    for task in eval_tasks:
        print("======================================================================")
        print("Without Few shot")
        start = time.time()

        try:
            result_no_fs = await run_without_few_shots(task["task"])
            time_no_fs = time.time() - start
            answer_no_fs = str(result_no_fs.final_result())
            steps_no_fs = len(result_no_fs.history)
            correct_no_fs = check_answer(answer_no_fs, task["expected_answer"])
        except Exception as e:
            time_no_fs = time.time() - start
            answer_no_fs = f"ERROR: {e}"
            steps_no_fs = None
            correct_no_fs = False

        print(f"  Answer: {answer_no_fs}")
        print(f"  Steps: {steps_no_fs}")
        print(f"  Time: {time_no_fs:.1f}s")
        print(f"  Correct: {correct_no_fs}")

        print("======================================================================")
        print("With Few Shots")
        start = time.time()
        try:
            result_fs = await run_with_few_shots(task["task"])
            time_fs = time.time() - start
            answer_fs = str(result_fs.final_result())
            steps_fs = len(result_fs.history)
            correct_fs = check_answer(answer_fs, task["expected_answer"])
        except Exception as e:
            time_fs = time.time() - start
            answer_fs = f"ERROR: {e}"
            steps_fs = -1
            correct_fs = False
        
        print(f"  Answer: {answer_fs}")
        print(f"  Steps: {steps_fs}")
        print(f"  Time: {time_fs:.1f}s")
        print(f"  Correct: {correct_fs}")


        results.append({
            "task": task["task"],
            "category": task["category"],
            "no_few_shot": {
                "answer": answer_no_fs,
                "steps": steps_no_fs,
                "time": round(time_no_fs, 1),
                "correct": correct_no_fs
            },
            "few_shot": {
                "answer": answer_fs,
                "steps": steps_fs,
                "time": round(time_fs, 1),
                "correct": correct_fs
            }
        })


    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    

if __name__ == "__main__":
    asyncio.run(evaluate())


