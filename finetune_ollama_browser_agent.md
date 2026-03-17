# Fine-Tuning an Ollama Model for Browser Automation

## End-to-End Pipeline

```
few_shot_examples.json
        │
        ▼
┌──────────────────┐
│ 1. Format Data   │  Convert to chat-completion format with system prompt
│    as JSONL      │  matching your agent's actual schema
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ 2. Fine-tune     │  Unsloth + LoRA on Google Colab (free T4 / A100)
│    with Unsloth  │  Base: unsloth/Qwen3-8B or Qwen2.5-7B-Instruct
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ 3. Export GGUF   │  Quantize to Q4_K_M or Q5_K_M for your 6GB RTX 4050
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ 4. Create        │  Write a Modelfile, `ollama create`, serve
│    Ollama Model  │
└──────────────────┘
        │
        ▼
┌──────────────────┐
│ 5. Evaluate &    │  Test on held-out tasks, collect real traces,
│    Iterate       │  add to training set, repeat
└──────────────────┘
```

---

## Step 1: Format Training Data

Your agent's real input isn't just the task string — it's the task + DOM state. Your training
data should mirror the exact prompt format your `browser-use` agent sends to the LLM.

### 1a. Convert few-shot examples to chat format

```python
# convert_to_training_data.py
import json

SYSTEM_PROMPT = """You are a browser automation agent. You receive a task and the current 
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
    """Convert a single few-shot example into multi-turn chat messages."""
    conversations = []
    
    # Simulate the first turn: user gives task + initial (empty) browser state
    user_msg = f"""Task: {task}

Current browser state:
URL: about:blank
Title: New Tab
Available elements:
[0] <input type="text" name="url" placeholder="Enter URL"/>
[1] <button>Go</button>
"""
    
    # Build the assistant's first action from step 0
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
                "evaluation_previous_goal": "Page loaded",
                "memory": f"Task: {task}",
                "next_goal": f"Click on {action_value}"
            },
            "action": [{"click_element": {"index": 0}}]  # placeholder index
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


def build_training_set(examples_file, output_file):
    with open(examples_file) as f:
        examples = json.load(f)
    
    training_data = []
    for ex in examples:
        for i, step in enumerate(ex["steps"]):
            convos = convert_example(ex["task"], ex["steps"][i:], turn_index=i)
            training_data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *convos
                ]
            })
    
    with open(output_file, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Generated {len(training_data)} training examples -> {output_file}")


if __name__ == "__main__":
    build_training_set("few_shot_examples_expanded.json", "training_data.jsonl")
```

### 1b. Better approach — Collect real agent traces

The synthetic conversion above is a starting point. For much better results, collect
real execution traces from your browser-use agent running with a strong model:

```python
# collect_traces.py — run with your Flask app + browser-use agent
# After each successful task, save the full trace

import json
from pathlib import Path

TRACE_DIR = Path("traces/")
TRACE_DIR.mkdir(exist_ok=True)

def save_trace(task: str, trace_steps: list[dict]):
    """
    Each trace_step should have:
    {
        "browser_state": { "url": "...", "title": "...", "elements": [...] },
        "llm_input": "the full prompt sent to LLM",
        "llm_output": "the raw JSON response from LLM",
        "action_taken": { "action_name": { "param": "value" } },
        "success": true/false
    }
    """
    trace = {"task": task, "steps": trace_steps}
    filename = TRACE_DIR / f"trace_{len(list(TRACE_DIR.glob('*.json'))):04d}.json"
    with open(filename, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Saved trace: {filename}")


def traces_to_training_data(trace_dir: str, output_file: str):
    """Convert collected traces into fine-tuning JSONL."""
    traces = sorted(Path(trace_dir).glob("*.json"))
    training_examples = []
    
    for trace_file in traces:
        with open(trace_file) as f:
            trace = json.load(f)
        
        # Only use successful traces
        if not all(s.get("success", True) for s in trace["steps"]):
            continue
        
        for step in trace["steps"]:
            training_examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": step["llm_input"]},
                    {"role": "assistant", "content": step["llm_output"]}
                ]
            })
    
    with open(output_file, "w") as f:
        for ex in training_examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Generated {len(training_examples)} examples from {len(traces)} traces")
```

---

## Step 2: Fine-Tune with Unsloth on Google Colab

Open a Colab notebook with a T4 (free) or A100 (Pro) GPU and run:

```python
# ============================================================
# Cell 1: Install Unsloth
# ============================================================
%%capture
!pip install unsloth
!pip install --force-reinstall --no-cache-dir --no-deps unsloth

# ============================================================
# Cell 2: Load base model with 4-bit quantization
# ============================================================
from unsloth import FastLanguageModel
import torch

max_seq_length = 4096  # browser DOM states can be long
dtype = None           # auto-detect
load_in_4bit = True    # fits on free Colab T4

model, tokenizer = FastLanguageModel.from_pretrained(
    # Pick ONE of these based on your needs:
    model_name = "unsloth/Qwen3-8B",              # latest Qwen3
    # model_name = "unsloth/Qwen2.5-7B-Instruct", # proven instruct model
    # model_name = "unsloth/Qwen3-4B",             # smaller, faster inference
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ============================================================
# Cell 3: Add LoRA adapters
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,              # LoRA rank — 16-64 is the sweet spot
    target_modules = [   # which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 32,     # usually set equal to r
    lora_dropout = 0,    # Unsloth optimized = 0
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # saves 60% VRAM
    random_state = 42,
)

# ============================================================
# Cell 4: Load and format training data
# ============================================================
from datasets import load_dataset

# Upload your training_data.jsonl to Colab first
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# Format into the chat template the model expects
def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_chat)
print(f"Training examples: {len(dataset)}")
print(f"Sample:\n{dataset[0]['text'][:500]}")

# ============================================================
# Cell 5: Configure training
# ============================================================
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True,  # pack short examples together for efficiency
    args = TrainingArguments(
        # --- Core hyperparameters ---
        learning_rate = 2e-4,
        num_train_epochs = 3,           # 3-5 epochs for <500 examples
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # effective batch size = 8
        
        # --- Optimization ---
        warmup_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        
        # --- Precision ---
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        
        # --- Logging ---
        logging_steps = 5,
        output_dir = "outputs",
        save_strategy = "epoch",
        seed = 42,
        report_to = "none",
    ),
)

# ============================================================
# Cell 6: Train!
# ============================================================
print("Starting fine-tuning...")
stats = trainer.train()
print(f"Training loss: {stats.training_loss:.4f}")

# ============================================================
# Cell 7: Quick sanity check before export
# ============================================================
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "You are a browser automation agent..."},  # your system prompt
    {"role": "user", "content": """Task: Find the price of RTX 4090 on Best Buy

Current browser state:
URL: about:blank
Title: New Tab
Available elements:
[0] <input type="text" placeholder="Search"/>
"""}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

output = model.generate(input_ids=inputs, max_new_tokens=256, temperature=0.1)
response = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
print(response)

# Verify it's valid JSON
import json
try:
    parsed = json.loads(response)
    print("✅ Valid JSON output")
    print(json.dumps(parsed, indent=2))
except json.JSONDecodeError as e:
    print(f"❌ Invalid JSON: {e}")
```

---

## Step 3: Export to GGUF

```python
# ============================================================
# Cell 8: Save as GGUF for Ollama
# ============================================================

# Option A: Quantized GGUF (recommended for your 6GB RTX 4050)
model.save_pretrained_gguf(
    "browser-agent-qwen3-8b",
    tokenizer,
    quantization_method = "q4_k_m",  # ~4.5GB, good quality/size balance
    # Other options:
    # "q5_k_m"  — ~5.5GB, better quality but tight on 6GB VRAM
    # "q3_k_m"  — ~3.5GB, if you need headroom for browser-use overhead
    # "q8_0"    — ~8GB, won't fit on your 4050 but good for CPU inference
)

# Option B: Full precision (if you want to quantize later or use on a bigger GPU)
# model.save_pretrained_merged("browser-agent-full", tokenizer, save_method="merged_16bit")

# ============================================================
# Cell 9: Download the GGUF file
# ============================================================
from google.colab import files
files.download("browser-agent-qwen3-8b/browser-agent-qwen3-8b-Q4_K_M.gguf")
# The filename pattern is: {dir_name}-{quantization}.gguf
```

---

## Step 4: Import into Ollama

On your local machine (Windows with RTX 4050 or Mac):

### 4a. Create a Modelfile

```dockerfile
# Modelfile
FROM ./browser-agent-qwen3-8b-Q4_K_M.gguf

# Template — must match the chat template used during fine-tuning
# For Qwen3 models:
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""

# System prompt baked in
SYSTEM """You are a browser automation agent. You receive a task and the current state of the browser (URL, page title, and visible elements with their indices).

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
- done: {"text": "extracted result"}"""

# Parameters tuned for structured output
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
PARAMETER stop <|im_end|>
```

### 4b. Create and test the model

```bash
# Create the model in Ollama
ollama create browser-agent -f Modelfile

# Quick test
ollama run browser-agent "Task: Find the price of MacBook Pro on Apple.com

Current browser state:
URL: about:blank
Title: New Tab
Available elements:
[0] <input type='text' placeholder='Search'/>
"

# List your models
ollama list
```

### 4c. Update your Flask app to use the new model

```python
# In your browser-use Flask app, change the model name:
# Before:
# model = "qwen3:8b"
# After:
model = "browser-agent"

# Everything else in your browser-use + Ollama integration stays the same
```

---

## Step 5: Evaluation & Iteration Loop

### 5a. Create a test set (hold out ~20 examples)

```python
# eval_agent.py
import json
import subprocess
import time

TEST_TASKS = [
    "Find the office hours for professor X at UMD",
    "Check the price of item Y on Amazon",
    "Find the submission deadline for conference Z",
    # ... 15-20 diverse tasks
]

def evaluate_model(model_name: str, tasks: list[str]):
    results = {"pass": 0, "fail": 0, "invalid_json": 0, "details": []}
    
    for task in tasks:
        try:
            # Run via ollama CLI or your agent's API
            response = subprocess.run(
                ["ollama", "run", model_name, f"Task: {task}\n\nCurrent browser state:\nURL: about:blank\nTitle: New Tab\nAvailable elements:\n[0] <input type='text'/>"],
                capture_output=True, text=True, timeout=30
            )
            output = response.stdout.strip()
            
            # Check if valid JSON
            parsed = json.loads(output)
            
            # Check required fields
            has_state = "current_state" in parsed
            has_action = "action" in parsed
            valid_action = isinstance(parsed.get("action"), list) and len(parsed["action"]) > 0
            
            if has_state and has_action and valid_action:
                results["pass"] += 1
                results["details"].append({"task": task, "status": "pass", "output": parsed})
            else:
                results["fail"] += 1
                results["details"].append({"task": task, "status": "fail", "reason": "missing fields"})
                
        except json.JSONDecodeError:
            results["invalid_json"] += 1
            results["details"].append({"task": task, "status": "invalid_json", "raw": output[:200]})
        except Exception as e:
            results["fail"] += 1
            results["details"].append({"task": task, "status": "error", "reason": str(e)})
    
    total = len(tasks)
    print(f"\n=== Evaluation Results for {model_name} ===")
    print(f"Pass:         {results['pass']}/{total} ({100*results['pass']/total:.1f}%)")
    print(f"Fail:         {results['fail']}/{total}")
    print(f"Invalid JSON: {results['invalid_json']}/{total}")
    
    return results

# Compare before and after fine-tuning
print("=== Baseline (Qwen3:8b) ===")
baseline = evaluate_model("qwen3:8b", TEST_TASKS)

print("\n=== Fine-tuned ===")
finetuned = evaluate_model("browser-agent", TEST_TASKS)
```

### 5b. Iteration strategy

```
Round 1: Fine-tune on 113 synthetic examples (format compliance)
         Expected: ~70-80% valid JSON output
         
Round 2: Run agent on 50 real tasks with the fine-tuned model
         Collect successful traces → add to training set
         Fix common failure patterns → add negative examples
         Re-train on ~300-500 examples
         Expected: ~85-90% valid JSON, better action selection
         
Round 3: Focus on hard cases (multi-step navigation, dynamic pages)
         Add traces with error recovery
         Train on ~500-1000 examples  
         Expected: ~90%+ valid JSON, robust action selection
```

---

## Key Tips for Your Setup

### VRAM Budget (RTX 4050 Laptop, 6GB)
| Quantization | Model Size | Remaining for Context | Recommended? |
|-------------|------------|----------------------|--------------|
| Q3_K_M      | ~3.5 GB    | ~2.5 GB              | If you need long DOM states |
| Q4_K_M      | ~4.5 GB    | ~1.5 GB              | ✅ Best balance |
| Q5_K_M      | ~5.5 GB    | ~0.5 GB              | Tight, may OOM with long pages |
| Q8_0        | ~8 GB      | Won't fit             | ❌ Use CPU only |

### Hyperparameter Cheat Sheet
| Parameter | Small dataset (<200) | Medium (200-1000) | Large (1000+) |
|-----------|---------------------|-------------------|---------------|
| Epochs    | 3-5                 | 2-3               | 1-2           |
| LoRA r    | 16-32               | 32-64             | 64            |
| LR        | 2e-4                | 1e-4              | 5e-5          |
| Batch size| 4-8 (effective)     | 8-16              | 16-32         |

### Common Pitfalls
1. **Chat template mismatch**: The Modelfile TEMPLATE must exactly match what `tokenizer.apply_chat_template()` produces during training. Mismatched templates = garbage output.
2. **Training on system prompt**: Make sure the system prompt is part of the training data, not just the user/assistant turns.
3. **Overfitting on small data**: With 113 examples, watch the training loss — if it drops below 0.1, you're likely overfitting. Use 3 epochs max.
4. **DOM truncation**: Real pages can have 500+ elements. Train with truncated DOM states (top 50 visible elements) to match what your agent actually sends.
5. **Thinking tokens**: Qwen3 models may produce `<think>...</think>` blocks before the JSON. Either train with `enable_thinking=False` in the chat template, or strip thinking tokens in your agent's response parser.
