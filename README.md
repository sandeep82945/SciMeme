# 🧠 SciMemeX: Automatic Generation of Scientific Memes from Research Articles

> **Paper**: *Breaking Bored? SciMemeX: Towards Automatic Generation of Scientific Memes from Research Articles*  
> **Framework**: Modular Multi-Agent System for Contrastive Meme Generation  
> **Key Components**: LLM-based Summarization, Template Selection, Meme Generation, and Evaluation

---

## 🌟 Overview

**SciMemeX** introduces a novel paradigm for **science communication** by automatically generating **scientific memes** from research papers.  
It leverages a **multi-agent framework** combining **idea extraction**, **template selection**, and **feedback-guided contrastive refinement** to produce memes that are:

- **Faithful** to the scientific content (scientific fidelity)  
- **Clear** in their communicative intent (clarity & interpretability)  
- **Engaging** for broader scientific audiences (engagement potential)

The system is designed to **translate dense academic insights into accessible, humorous, and memorable visual narratives**, fostering wider engagement beyond academia.

---

## 🧩 Core Components

| Module | Description |
|--------|--------------|
| **Concisio Agent** | Extracts comparative insights: *prior work vs. new contributions* (~100-word summary). |
| **Template Selector Agent (TSA)** | Chooses optimal meme templates for given paper ideas. |
| **Generator Agent** | Produces creative meme captions conditioned on paper content and selected templates. |
| **Contrastive Generator** | Refines memes iteratively using feedback from prior best/worst generations. |
| **Evaluation Agents** | Three LLM-as-judge evaluators for *Fidelity*, *Clarity*, and *Engagement*. |

---

## 🧠 Evaluation Dimensions

SciMemeX employs three complementary automatic evaluation metrics implemented in `eval_metrics.py`:

| Metric | Scale | Definition |
|---------|--------|------------|
| **Scientific Fidelity** | 1–5 | Faithfulness to the paper’s core contribution. |
| **Clarity (FRI Score)** | 1–3 | How easily a graduate-level audience grasps the meme’s message. |
| **Engagement Potential** | 1–5 | Humor, shareability, and non-offensiveness. |

---

### 2️⃣ Set Up Virtual Environment
```
python3 -m venv scimemex_env
source scimemex_env/bin/activate```


### Set Up Virtual Environment
```pip install -r requirements.txt```


### 🔧 Configuration
Edit config.yaml as follows:

api_keys:
  openai: "YOUR_OPENAI_API_KEY"

models:
  default: gpt-4o

temperatures:
  fidelity: 0.2
  clarity: 0.5
  engagement: 0.2

data:
  path: "./papers_json/"
  amount: 1000

---
### 🚀 Usage
```
python eval/eval_run.py
```

Run Full Multi-Agent Meme Generation

```python SciMemeX.py --config_path config.yaml --output results/output.json```


### 🧾 Example Output

{
  "paper_id": "2305.12345",
  "final_best_meme": "Using recurrence for sequence modeling → Using self-attention for everything.",
  "final_best_score": 10.82,
  "iteration_scores": [
    {"iteration": 1, "fidelity": 4.5, "clarity": 2.1, "engagement": 4.0, "average": 10.6}
  ]
}




