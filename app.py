from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from browser_use import Browser, SystemPrompt
from browser_use.agent.service import Agent
from browser_use.agent.views import AgentOutput
from browser_use.browser.browser import BrowserConfig
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from collections import deque
from threading import Lock
import random
from few_shot import FEW_SHOT_EXAMPLES
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from dotenv import load_dotenv
import os
from starlette.responses import JSONResponse, RedirectResponse
import requests
import nest_asyncio
nest_asyncio.apply()
import asyncio
from datetime import datetime, timezone
from urllib.parse import quote, quote_plus
from pymongo import MongoClient
from pymongo.errors import OperationFailure


class CustomSystemPrompt(SystemPrompt):
    def __init__(self, *args, task_examples="", **kwargs):
        super().__init__(*args, **kwargs)
        self.task_examples = task_examples


    def important_rules(self):
        existing = super().important_rules()
        return existing + "\n" + SYSTEM_PROMPT + "\n" + self.task_examples


def _env_str(key: str) -> str:
    v = os.getenv(key)
    return (v or "").strip()


def _mongo_connection_uri() -> str | None:
    user = _env_str("MONGODB_USERNAME")
    password = _env_str("MONGODB_PASSWORD")
    host = _env_str("MONGODB_HOST")
    if user and password and host:
        return (
            f"mongodb+srv://{quote_plus(user)}:{quote_plus(password)}@{host}/"
            "?retryWrites=true&w=majority"
        )
    uri = _env_str("MONGODB_CONNECTION_STRING")
    return uri or None


def upsert_user_from_oauth(user_info: dict) -> bool:
    if users_collection is None:
        raise RuntimeError(
            "MongoDB is not configured. Set MONGODB_USERNAME, MONGODB_PASSWORD, and "
            "MONGODB_HOST, or set MONGODB_CONNECTION_STRING."
        )

    email = (user_info.get("email") or "").strip()
    if not email:
        raise ValueError("Google userinfo did not include an email")

    google_sub = user_info.get("sub")
    name = user_info.get("name") or ""
    picture = user_info.get("picture") or ""

    now = datetime.now(timezone.utc)
    try:
        result = users_collection.update_one(
            {"email": email},
            {
                "$setOnInsert": {
                    "email": email,
                    "google_sub": google_sub,
                    "name": name,
                    "picture": picture,
                    "created_at": now,
                }
            },
            upsert=True,
        )
    except OperationFailure as exc:
        err = str(exc).lower()
        if getattr(exc, "code", None) == 8000 or "bad auth" in err or "authentication failed" in err:
            raise RuntimeError(
                "MongoDB Atlas rejected the database username or password. In Atlas, open "
                "Database Access → your user → Edit Password, then update MONGODB_PASSWORD "
                "(and MONGODB_CONNECTION_STRING if you use it). If the password contains "
                "@ : / ? # % or spaces, use MONGODB_USERNAME, MONGODB_PASSWORD, and "
                "MONGODB_HOST in .env so the app can encode them correctly."
            ) from exc
        raise

    return result.upserted_id is not None


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


async def run_agent(task):
    with memory_lock:
        live_memory.clear()

    browser = Browser(config=BrowserConfig(headless=True))

    selected_few_shots = select_examples(task, 3, True)
    few_shot_string = format_few_shot_examples(selected_few_shots)

    live_traces = []
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

    return task, result, agent, live_traces


app = FastAPI()

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are a people-finding assistant for cold outreach. Follow these rules:

1. Go to Google and search for the person, role, or company.
2. Look for LinkedIn profiles, company team pages, or about pages.
3. Extract names, titles, and email patterns.
4. Present results clearly and say done. Do NOT over-browse.
"""

llm = ChatOllama(model="kimi-k2.5:cloud", temperature=0.4)
live_memory = deque(maxlen=100)
memory_lock = Lock()
load_dotenv()

CLIENT_SECRET_FILE = os.getenv("CLIENT_SECRET_FILE")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SCOPES = ['openid', 'https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile']

oauth_states = {}

_mongo_uri = _mongo_connection_uri()
mongo_client = (
    MongoClient(_mongo_uri, serverSelectionTimeoutMS=10_000) if _mongo_uri else None
)
mongo_db = mongo_client["reacher"] if mongo_client else None
users_collection = mongo_db["users"] if mongo_db else None





@app.get("/")
def root():
    return {"Hello World"}


@app.get("/callback")
async def callback(code, state):
    # if request.args.get('state') != session['state']:
    #     raise Exception('Invalid state')

    if state != oauth_states.get('state'):
        return {"error": "Invalid state"}
    
    flow = Flow.from_client_secrets_file(CLIENT_SECRET_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = "http://localhost:8000/callback"
    flow.fetch_token(code=code)

    credentials = flow.credentials

    user_info = requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {credentials.token}"}
    ).json()

    name = user_info.get("name", "")
    email = user_info.get("email", "")

    try:
        is_new = upsert_user_from_oauth(user_info)
    except (RuntimeError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    q = (
        f"name={quote(name)}&email={quote(email)}&is_new={'true' if is_new else 'false'}"
    )
    return RedirectResponse(f"http://localhost:5173/?{q}")


@app.get("/auth/google")
def google_login():
    flow = Flow.from_client_secrets_file(CLIENT_SECRET_FILE, scopes=SCOPES)
    flow.redirect_uri = "http://localhost:8000/callback"
    authorization_url, state = flow.authorization_url(access_type='offline', prompt='select_account')
    oauth_states['state'] = state

    return RedirectResponse(authorization_url)


@app.post("/processQuery")
async def process_query(query: str):
    print(query)

    task, result, agent, trace_steps = asyncio.run(run_agent(query))
    print(result.final_result())

    # return {"result": "Processing query"}
    return JSONResponse({"result": result.final_result(), "trace": trace_steps})
