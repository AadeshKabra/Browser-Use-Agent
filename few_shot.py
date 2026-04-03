import json
import random


FEW_SHOT_EXAMPLES = {
 
    # ── CATEGORY 1: Find specific people at a company ──
    "extract_info": [
        {
            "task": "Find the email of Anthropic's CTO",
            "steps": [
                {"action": "go_to_url", "url": "https://www.google.com"},
                {"action": "input_text", "target": "search box", "text": "Anthropic CTO email LinkedIn"},
                {"action": "click_element", "target": "most relevant result"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Anthropic's CTO is Tom Brown. Email pattern at Anthropic is first@anthropic.com, so likely tom@anthropic.com"}
            ]
        },
        {
            "task": "Get the email of the head of engineering at Stripe",
            "steps": [
                {"action": "go_to_url", "url": "https://www.google.com"},
                {"action": "input_text", "target": "search box", "text": "Stripe head of engineering LinkedIn"},
                {"action": "click_element", "target": "most relevant LinkedIn result"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Found: David Singleton, CTO at Stripe. LinkedIn profile confirmed."}
            ]
        },
    ],
 
    # ── CATEGORY 2: Search for roles/people at companies ──
    "search": [
        {
            "task": "Find ML engineers at Nvidia",
            "steps": [
                {"action": "go_to_url", "url": "https://www.google.com"},
                {"action": "input_text", "target": "search box", "text": "ML engineers Nvidia LinkedIn"},
                {"action": "click_element", "target": "LinkedIn search result"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Found 5 ML engineers at Nvidia: [names and titles listed]"}
            ]
        },
        {
            "task": "Find backend engineers at Shopify who work on payments",
            "steps": [
                {"action": "go_to_url", "url": "https://www.google.com"},
                {"action": "input_text", "target": "search box", "text": "Shopify backend engineer payments team LinkedIn"},
                {"action": "click_element", "target": "most relevant LinkedIn result"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Found 3 backend engineers on Shopify's payments team with LinkedIn profiles"}
            ]
        },
    ],
 
    # ── CATEGORY 3: Find hiring managers / recruiters ──
    "navigation": [
        {
            "task": "Find who leads recruiting at Databricks",
            "steps": [
                {"action": "go_to_url", "url": "https://www.google.com"},
                {"action": "input_text", "target": "search box", "text": "Databricks head of recruiting LinkedIn"},
                {"action": "click_element", "target": "LinkedIn result"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Found: VP of Talent Acquisition at Databricks with LinkedIn profile"}
            ]
        },
        {
            "task": "Find the engineering manager for the AI team at Notion",
            "steps": [
                {"action": "go_to_url", "url": "https://www.google.com"},
                {"action": "input_text", "target": "search box", "text": "Notion AI team engineering manager LinkedIn"},
                {"action": "click_element", "target": "most relevant result"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Found the engineering manager for Notion's AI team on LinkedIn"}
            ]
        },
    ],
 
    # ── CATEGORY 4: Company research for outreach ──
    "comparison": [
        {
            "task": "Find cybersecurity startups hiring in DC",
            "steps": [
                {"action": "go_to_url", "url": "https://www.google.com"},
                {"action": "input_text", "target": "search box", "text": "cybersecurity startups hiring Washington DC 2026"},
                {"action": "click_element", "target": "most relevant result"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Found 4 cybersecurity startups actively hiring in DC: [company names and roles]"}
            ]
        },
        {
            "task": "Find AI startups in San Francisco that recently raised Series A",
            "steps": [
                {"action": "go_to_url", "url": "https://www.google.com"},
                {"action": "input_text", "target": "search box", "text": "AI startups San Francisco Series A 2026"},
                {"action": "click_element", "target": "most relevant result"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Found 3 AI startups with recent Series A: [company names, funding amounts]"}
            ]
        },
    ],
 
    # ── CATEGORY 5: Error recovery ──
    "error_recovery": [
        {
            "task": "Find the VP of Engineering at a stealth startup with no website",
            "steps": [
                {"action": "go_to_url", "url": "https://www.google.com"},
                {"action": "input_text", "target": "search box", "text": "stealth startup VP engineering LinkedIn"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Could not find specific results for a stealth startup. Recommend searching LinkedIn directly with company name if known."}
            ]
        },
    ],
}


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


def format_examples_for_prompt(examples: list) -> str:
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


if __name__ == "__main__":
    
    # with open("few_shot_examples.json", "w") as f:
        # json.dump(FEW_SHOT_EXAMPLES, f, indent=2)
    # print(f"Saved {len(FEW_SHOT_EXAMPLES)} examples to few_shot_examples.json")

    test_queries = [
        "Find the email address of professor Zhicheng Liu at UMD",
        "Search for machine learning repos on GitHub",
        "Fill out the application form for the internship",
        "Compare AWS Lambda vs Google Cloud Functions pricing",
        "What's the latest blog post on Anthropic's website",
    ]

    for query in test_queries:
        category = classify_task(query)
        examples = select_examples(query)

        print(f"Query: {query}")
        print(f"  → Category: {category}")
        print(f"  → Examples selected: {[e['task'] for e in examples]}")
        print(f"\nFormatted prompt:\n{format_examples_for_prompt(examples)}")
        print("=" * 60)

    
    all_examples = []
    for cat_example in FEW_SHOT_EXAMPLES.values():
        all_examples.extend(cat_example)