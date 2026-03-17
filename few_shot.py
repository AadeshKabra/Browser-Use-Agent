# from datasets import load_dataset

# ds = load_dataset("McGill-NLP/WebLINX", "weblinx-full", split="train", streaming=True)

# for i, item in enumerate(ds):
#     print(item.keys())
#     break

# import json


# FEW_SHOT_EXAMPLES = [
#   {
#     "task": "Find the research lab run by professor Abhinav Bhatele at UMD",
#     "steps": [
#       "go_to_url: https://www.cs.umd.edu/people/faculty",
#       "click_element: Abhinav Bhatele",
#       "done: Parallel Software and Systems Group"
#     ]
#   },
#   {
#     "task": "Find the office location of professor Hal Daume at UMD CS",
#     "steps": [
#       "go_to_url: https://www.cs.umd.edu/people/faculty",
#       "click_element: Hal Daume",
#       "done: Extract office number from profile"
#     ]
#   },
#   {
#     "task": "What are the admission requirements for UMD CS PhD program",
#     "steps": [
#       "go_to_url: https://www.cs.umd.edu/grad/catalog",
#       "click_element: Admission Requirements or PhD Program",
#       "done: Extract admission requirements"
#     ]
#   },
#   {
#     "task": "Find the latest research papers published by Stanford NLP group",
#     "steps": [
#       "go_to_url: https://nlp.stanford.edu/pubs/",
#       "done: Extract list of recent publications"
#     ]
#   },
#   {
#     "task": "What is the price of iPhone 16 Pro on Apple's website",
#     "steps": [
#       "go_to_url: https://www.apple.com/shop/buy-iphone",
#       "click_element: iPhone 16 Pro",
#       "done: Extract starting price"
#     ]
#   },
#   {
#     "task": "Find the hours of operation for the McKeldin Library at UMD",
#     "steps": [
#       "go_to_url: https://www.lib.umd.edu/visit/hours",
#       "done: Extract McKeldin Library hours"
#     ]
#   },
#   {
#     "task": "Check the current weather in College Park, Maryland",
#     "steps": [
#       "go_to_url: https://weather.com",
#       "input_text: College Park, MD in search box",
#       "click_element: Search or College Park result",
#       "done: Extract current temperature and conditions"
#     ]
#   },
#   {
#     "task": "Find the top trending repositories on GitHub today",
#     "steps": [
#       "go_to_url: https://github.com/trending",
#       "done: Extract list of trending repositories"
#     ]
#   },
#   {
#     "task": "Look up the schedule for UMD CS course CMSC828A",
#     "steps": [
#       "go_to_url: https://app.testudo.umd.edu/soc/",
#       "input_text: CMSC in search box",
#       "click_element: Search",
#       "click_element: CMSC828A",
#       "done: Extract course schedule and instructor"
#     ]
#   },
#   {
#     "task": "Find how many citations a paper titled 'Attention is All You Need' has on Google Scholar",
#     "steps": [
#       "go_to_url: https://scholar.google.com",
#       "input_text: Attention is All You Need",
#       "click_element: Search",
#       "done: Extract citation count from result"
#     ]
#   },
#   {
#     "task": "Find the menu for a pizza restaurant near College Park MD",
#     "steps": [
#       "go_to_url: https://www.google.com/maps",
#       "input_text: pizza restaurant College Park MD",
#       "click_element: First restaurant result",
#       "click_element: Menu",
#       "done: Extract menu items and prices"
#     ]
#   },
#   {
#     "task": "What is the deadline for NeurIPS 2025 paper submission",
#     "steps": [
#       "go_to_url: https://neurips.cc",
#       "click_element: Dates or Call for Papers",
#       "done: Extract submission deadline"
#     ]
#   },
#   {
#     "task": "Find the return policy on Amazon",
#     "steps": [
#       "go_to_url: https://www.amazon.com/gp/help/customer/display.html",
#       "click_element: Returns and Refunds",
#       "done: Extract return policy details"
#     ]
#   },
#   {
#     "task": "Look up the bus schedule for Route 104 from UMD campus",
#     "steps": [
#       "go_to_url: https://www.transportation.umd.edu/shuttle",
#       "click_element: Route 104",
#       "done: Extract schedule and stops"
#     ]
#   },
#   {
#     "task": "Find the LinkedIn profile of a company called Modular AI",
#     "steps": [
#       "go_to_url: https://www.linkedin.com/company/modular-ai",
#       "done: Extract company description and details"
#     ]
#   }
# ]


import json
import random


FEW_SHOT_EXAMPLES = {
 
    # ── CATEGORY 1: Information Extraction (find a specific fact on a page) ──
    "extract_info": [
        {
            "task": "Find the email address of professor Zhicheng Liu at UMD CS",
            "steps": [
                {"action": "go_to_url", "url": "https://www.cs.umd.edu/people/faculty"},
                {"action": "scroll_down", "amount": 3},
                {"action": "click_element", "target": "Zhicheng Liu"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Professor Zhicheng Liu's email is zliu@umd.edu"}
            ]
        },
        {
            "task": "What is the price of the MacBook Air M4 on Apple's website?",
            "steps": [
                {"action": "go_to_url", "url": "https://www.apple.com/shop/buy-mac/macbook-air"},
                {"action": "click_element", "target": "MacBook Air M4"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "The MacBook Air M4 starts at $1,099"}
            ]
        },
        {
            "task": "How many citations does 'Attention is All You Need' have on Google Scholar?",
            "steps": [
                {"action": "go_to_url", "url": "https://scholar.google.com"},
                {"action": "input_text", "target": "search box", "text": "Attention is All You Need"},
                {"action": "click_element", "target": "Search button"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "The paper 'Attention is All You Need' has approximately 120,000 citations"}
            ]
        },
        {
            "task": "Find the return policy for electronics on Best Buy",
            "steps": [
                {"action": "go_to_url", "url": "https://www.bestbuy.com/site/help-topics/return-exchange-policy/pcmcat260800050014.c"},
                {"action": "scroll_down", "amount": 2},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Best Buy has a 15-day return policy for most electronics, 30 days for My Best Buy Plus and Total members"}
            ]
        },
    ],
 
    # ── CATEGORY 2: Search & Navigate (find something via a search box) ──
    "search": [
        {
            "task": "Search for 'browser automation' on GitHub and find the most starred repo",
            "steps": [
                {"action": "go_to_url", "url": "https://github.com/search"},
                {"action": "input_text", "target": "search box", "text": "browser automation"},
                {"action": "click_element", "target": "Search button"},
                {"action": "click_element", "target": "Sort: Most stars"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "The most starred browser automation repo is playwright with 68k+ stars"}
            ]
        },
        {
            "task": "Look up the schedule for CMSC828A on UMD Testudo",
            "steps": [
                {"action": "go_to_url", "url": "https://app.testudo.umd.edu/soc/"},
                {"action": "input_text", "target": "search box", "text": "CMSC828A"},
                {"action": "click_element", "target": "Search"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "CMSC828A meets TuTh 2:00-3:15pm in IRB 0318, taught by Prof. Smith"}
            ]
        },
        {
            "task": "Find a highly rated Italian restaurant in downtown Chicago on Yelp",
            "steps": [
                {"action": "go_to_url", "url": "https://www.yelp.com"},
                {"action": "input_text", "target": "search box", "text": "Italian restaurant"},
                {"action": "input_text", "target": "location box", "text": "Downtown Chicago, IL"},
                {"action": "click_element", "target": "Search"},
                {"action": "scroll_down", "amount": 1},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Top rated: RPM Italian (4.5 stars, 3000+ reviews) at 52 W Illinois St"}
            ]
        },
        {
            "task": "Find Python jobs in New York on LinkedIn",
            "steps": [
                {"action": "go_to_url", "url": "https://www.linkedin.com/jobs/"},
                {"action": "input_text", "target": "search box", "text": "Python developer"},
                {"action": "input_text", "target": "location box", "text": "New York, NY"},
                {"action": "click_element", "target": "Search"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Found 500+ Python developer jobs in New York. Top result: Python Backend Engineer at Bloomberg, posted 2 days ago"}
            ]
        },
    ],
 
    # ── CATEGORY 3: Form Filling / Input-heavy tasks ──
    "form_fill": [
        {
            "task": "Sign up for a free trial on a SaaS website",
            "steps": [
                {"action": "go_to_url", "url": "https://example-saas.com/signup"},
                {"action": "input_text", "target": "Name field", "text": "John Doe"},
                {"action": "input_text", "target": "Email field", "text": "john@example.com"},
                {"action": "input_text", "target": "Password field", "text": "SecurePass123"},
                {"action": "click_element", "target": "Start Free Trial button"},
                {"action": "done", "text": "Successfully submitted the sign-up form"}
            ]
        },
        {
            "task": "Fill out a contact form to ask about bulk pricing",
            "steps": [
                {"action": "go_to_url", "url": "https://example-store.com/contact"},
                {"action": "input_text", "target": "Name field", "text": "Jane Smith"},
                {"action": "input_text", "target": "Email field", "text": "jane@company.com"},
                {"action": "input_text", "target": "Subject field", "text": "Bulk pricing inquiry"},
                {"action": "input_text", "target": "Message field", "text": "I'd like to inquire about bulk pricing for 500+ units."},
                {"action": "select_dropdown", "target": "Reason for contact", "value": "Sales"},
                {"action": "click_element", "target": "Submit"},
                {"action": "done", "text": "Contact form submitted successfully"}
            ]
        },
    ],
 
    # ── CATEGORY 4: Navigation / Multi-page browsing ──
    "navigation": [
        {
            "task": "Find the latest blog post on OpenAI's website",
            "steps": [
                {"action": "go_to_url", "url": "https://openai.com/blog"},
                {"action": "extract_page_content"},
                {"action": "click_element", "target": "first blog post title"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Latest blog post: 'Introducing GPT-5' published on March 10, 2026"}
            ]
        },
        {
            "task": "Find the submission deadline for NeurIPS 2026",
            "steps": [
                {"action": "go_to_url", "url": "https://neurips.cc"},
                {"action": "click_element", "target": "Dates or Call for Papers"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "NeurIPS 2026 abstract deadline: May 15, 2026. Full paper deadline: May 22, 2026"}
            ]
        },
        {
            "task": "Find what programming languages are used at Stripe from their jobs page",
            "steps": [
                {"action": "go_to_url", "url": "https://stripe.com/jobs"},
                {"action": "click_element", "target": "Engineering"},
                {"action": "scroll_down", "amount": 2},
                {"action": "click_element", "target": "a backend engineering role"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Stripe primarily uses Ruby, Java, and Go for backend engineering"}
            ]
        },
        {
            "task": "Check the trending topics on Reddit today",
            "steps": [
                {"action": "go_to_url", "url": "https://www.reddit.com"},
                {"action": "click_element", "target": "Popular"},
                {"action": "scroll_down", "amount": 2},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Top trending topics today include: AI regulation debate, NBA playoff results, and a viral cat video"}
            ]
        },
    ],
 
    # ── CATEGORY 5: Comparison / Multi-step research ──
    "comparison": [
        {
            "task": "Compare the pricing of ChatGPT Plus vs Claude Pro",
            "steps": [
                {"action": "go_to_url", "url": "https://openai.com/chatgpt/pricing/"},
                {"action": "extract_page_content"},
                {"action": "go_to_url", "url": "https://claude.ai/pricing"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "ChatGPT Plus costs $20/month. Claude Pro costs $20/month. Both offer similar pricing for individual plans."}
            ]
        },
        {
            "task": "Which has more GitHub stars: FastAPI or Flask?",
            "steps": [
                {"action": "go_to_url", "url": "https://github.com/tiangolo/fastapi"},
                {"action": "extract_page_content"},
                {"action": "go_to_url", "url": "https://github.com/pallets/flask"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "FastAPI has ~80k stars, Flask has ~68k stars. FastAPI has more GitHub stars."}
            ]
        },
    ],
 
    # ── CATEGORY 6: Error recovery / Fallback patterns ──
    "error_recovery": [
        {
            "task": "Find the office hours for a professor whose page returns a 404",
            "steps": [
                {"action": "go_to_url", "url": "https://www.cs.umd.edu/~oldprofessor"},
                {"action": "extract_page_content"},
                {"action": "go_to_url", "url": "https://www.google.com"},
                {"action": "input_text", "target": "search box", "text": "professor oldprofessor UMD CS office hours"},
                {"action": "click_element", "target": "most relevant result"},
                {"action": "extract_page_content"},
                {"action": "done", "text": "Found office hours via Google search: Mon/Wed 2-3pm, AVW 3245"}
            ]
        },
        {
            "task": "Find a sold-out product's expected restock date",
            "steps": [
                {"action": "go_to_url", "url": "https://store.example.com/product/widget-pro"},
                {"action": "extract_page_content"},
                {"action": "scroll_down", "amount": 2},
                {"action": "click_element", "target": "Notify me when available"},
                {"action": "done", "text": "Product is sold out. No restock date listed, but 'Notify me' option is available on the page."}
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