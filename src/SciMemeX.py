import os
import json
import yaml
import argparse
import random
import re
from autogen import ConversableAgent, LLMConfig
from utils.read_json import read_json
from eval.eval_run import evaluate_meme, get_score

# ------------------------------
# Parse config
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', required=True, help='Enter path of config file name')
parser.add_argument('--output', required=True, help='Enter path to save result')
args = parser.parse_args()

with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

amount = config['data']['amount']
openai_key = config['api_keys']['openai']
llm_config_model = 'gpt-4o'

gpt4o = LLMConfig(
    model=llm_config_model,
    api_key=openai_key,
)

# ------------------------------
# Load meme template dataset
# ------------------------------
with open("data/meme_template_dataset.json", "r") as f:
    meme_templates = json.load(f)

# ------------------------------
# Helper functions for robust JSON parsing
# ------------------------------
def extract_json_from_text(text):
    """Extract JSON from text that might contain additional content."""
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None

def safe_json_parse(text, fallback_data):
    """Safely parse JSON with fallback options."""
    if not text or not text.strip():
        print("Warning: Empty response received")
        return fallback_data
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    extracted = extract_json_from_text(text)
    if extracted:
        return extracted
    print(f"Warning: Could not parse JSON from response: {text[:200]}...")
    return fallback_data

# ------------------------------
# Paper Analysis Agents
# ------------------------------
def extract_paper_summary(full_text):
    """
    Extract core ideas and contrasts from full paper text using two specialized agents.
    Returns a concise ~100-word summary of prior work vs. new contributions.
    """
    llm_config = LLMConfig(
        model=llm_config_model,
        api_key=openai_key,
    )

    Innovation_Insight = ConversableAgent(
        name="Innovation_Insight",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message=(
            "You are a comparative research analyst. Your task is to deeply analyze a research paper "
            "to extract the main conceptual and technical differences between past work and this paper's contributions.\n\n"
            "Structure your response as:\n"
            "Prior Work (Old Ideas): Summarize how this problem was tackled in previous literature. Include model names, strategies, or limitations if possible.\n"
            "Proposed Approach (New Idea): What method, model, or strategy is introduced by this paper?\n"
            "Core Differences: A comparative analysis answering:\n"
            "- What *exactly* changes in methodology, architecture, or objective?\n"
            "- Why is this a meaningful improvement?\n"
            "- Are there tradeoffs or assumptions introduced?\n\n"
            "Be precise, technical, and avoid generic summaries.\n"
            "If you receive a summary from another agent then output it as it is without changing anything or providing feedback that would initiate another conversation"
        )
    )

    Concisio = ConversableAgent(
        name="Concisio",
        code_execution_config=False,
        human_input_mode="NEVER",
        llm_config=llm_config,
        system_message=(
            "Your task is to read a structured comparison between prior work and a new method, "
            "and produce a single paragraph that captures the **essential difference** in approach or idea.\n\n"
            "Your output should:\n"
            "- Clearly state what *was* done before\n"
            "- Explain what *this* paper does differently\n"
            "- Highlight why this difference is important\n"
            "- Be technical but readable in under 100 words\n\n"
            "Avoid vagueness. Do not invent anything not present in the input. Avoid summarizing the paper just state the core ideas in around 100 words.\n"
            "Avoid using filler text and only output the summary, avoid using lines such as 'here is your summary' etc."
        )
    )

    try:
        # First conversation: Concisio asks Innovation_Insight to analyze the paper
        Concisio.initiate_chat(
            Innovation_Insight,
            message=f'''
                This is content extracted from the research paper:\n\n{full_text}\n\n
                Your final output should be around 100 words only, capturing the key ideas and differences in the paper.
            ''',
            max_turns=1
        )

        # Extract the analysis from Innovation_Insight
        ideas = Concisio.chat_messages
        analysis = ideas[Innovation_Insight][-1]['content']

        # Second conversation: Innovation_Insight asks Concisio to condense
        Innovation_Insight.initiate_chat(
            Concisio,
            message=f'''
                This is summary extracted from the research paper:\n\n{analysis}\n\n
                Your final output should be around 100 words only, capturing the key ideas and differences in the paper.
            ''',
            max_turns=1
        )

        # Extract the final concise summary
        final_ideas = Innovation_Insight.chat_messages
        paper_summary = final_ideas[Concisio][-1]['content']

        return paper_summary

    except Exception as e:
        print(f"Error in paper analysis: {e}")
        # Return a fallback that at least contains the beginning of the paper
        return full_text[:1000] + "..."

# ------------------------------
# Define agents
# ------------------------------

# Generator agent
generator = ConversableAgent(
    name="generator",
    system_message=(
        "You are a creative meme generator specializing in academic research. "
        "Generate diverse, creative memes that contrast prior work vs. new contributions. "
        "Return only the meme template and text in a structured format."
    ),
    llm_config=gpt4o,
    human_input_mode="NEVER",
)

# Selector agent for choosing templates
meme_selector = ConversableAgent(
    name="meme_selector",
    system_message=(
        "You are a meme strategy selector. "
        "Given a paper's key ideas and a list of meme templates, "
        "choose the top-k templates that would most effectively contrast prior work vs. contributions. "
        "You MUST return your response as valid JSON in exactly this format: "
        '{"selected_templates": ["template_name_1", "template_name_2"], "justification": "your reasoning here"}. '
        "Do not include any other text or explanation outside the JSON."
    ),
    llm_config=gpt4o,
    human_input_mode="NEVER",
)

# ------------------------------
# Meme Selection + Generation
# ------------------------------

def select_meme_templates(paper_content, all_templates, k=5):
    template_list = [f"{name}: {info['description']}" for name, info in all_templates.items()]
    template_names = list(all_templates.keys())

    prompt = (
        f"Paper Content:\n{paper_content}\n\n"
        f"Here are {len(template_list)} meme templates:\n"
        + "\n".join([f"{i+1}. {t}" for i, t in enumerate(template_list)]) +
        f"\n\nTask: Pick the top {k} templates that would be most effective for this paper. "
        "Return ONLY valid JSON with keys: selected_templates (list of template names), justification (string)."
        f"\n\nAvailable template names: {template_names}"
    )

    try:
        # Initiate chat; using meme_selector with itself to get a single-agent response
        result = meme_selector.initiate_chat(meme_selector, message=prompt, max_turns=1)
        response_content = result.chat_history[-1]['content']  # relies on autogen's return format

        fallback_templates = random.sample(template_names, min(k, len(template_names)))
        fallback_data = {
            "selected_templates": fallback_templates,
            "justification": f"Randomly selected {k} templates due to parsing error"
        }

        parsed_result = safe_json_parse(response_content, fallback_data)

        if not isinstance(parsed_result.get('selected_templates'), list):
            print("Warning: selected_templates is not a list, using fallback")
            return fallback_data

        valid_templates = [t for t in parsed_result['selected_templates'] if t in template_names]
        if not valid_templates:
            print("Warning: No valid templates found, using fallback")
            return fallback_data

        parsed_result['selected_templates'] = valid_templates[:k]
        return parsed_result

    except Exception as e:
        print(f"Error in select_meme_templates: {e}")
        fallback_templates = random.sample(template_names, min(k, len(template_names)))
        return {
            "selected_templates": fallback_templates,
            "justification": f"Error occurred during template selection: {e}"
        }

def generate_memes_from_templates(paper_content, template_names):
    memes = []
    for template_name in template_names:
        try:
            full_prompt = (
                f"Paper Content:\n{paper_content}\n\n"
                f"Task: Create a meme using the '{template_name}' template. "
                "Make it specific to this paper's contribution vs prior work. "
                "Return only: Meme template and the meme text in structured format."
            )
            result = generator.initiate_chat(generator, message=full_prompt, max_turns=1)
            meme_content = result.chat_history[-1]['content']
            memes.append(meme_content)
        except Exception as e:
            print(f"Error generating meme with template {template_name}: {e}")
            memes.append(f"Template: {template_name}\nError generating meme: {e}")
    return memes

def generate_contrastive_memes(paper_content, best_meme, worst_meme, best_eval, worst_eval, chosen_templates):
    """
    Generate contrastive memes with evaluation reasoning.
    """
    memes = []
    for template_name in chosen_templates:
        try:
            best_reasoning = (
                f"Fidelity: {best_eval.get('fidelity', {}).get('reasoning', 'N/A')}\n"
                f"Clarity: {best_eval.get('clarity', {}).get('justification', 'N/A')}\n"
                f"Engagement: {best_eval.get('engagement', {}).get('reasoning', 'N/A')}"
            )

            worst_reasoning = (
                f"Fidelity: {worst_eval.get('fidelity', {}).get('reasoning', 'N/A')}\n"
                f"Clarity: {worst_eval.get('clarity', {}).get('justification', 'N/A')}\n"
                f"Engagement: {worst_eval.get('engagement', {}).get('reasoning', 'N/A')}"
            )

            prompt = (
                f"Paper Content:\n{paper_content}\n\n"
                f"Best meme so far:\n{best_meme}\n\n"
                f"Why it's good:\n{best_reasoning}\n\n"
                f"Worst meme so far:\n{worst_meme}\n\n"
                f"Why it's bad:\n{worst_reasoning}\n\n"
                f"Task: Create a meme using the '{template_name}' template that keeps the strengths of the best meme, "
                f"but avoids the weaknesses of the worst meme. "
                "Focus on improving fidelity, clarity, and engagement. "
                "Return only: Meme template and the meme text."
            )
            result = generator.initiate_chat(generator, message=prompt, max_turns=1)
            meme_content = result.chat_history[-1]['content']
            memes.append(meme_content)
        except Exception as e:
            print(f"Error generating contrastive meme with template {template_name}: {e}")
            memes.append(f"Template: {template_name}\nError generating contrastive meme: {e}")
    return memes

def evaluate_and_rank_memes(paper_content, memes):
    evaluated_memes = []
    for i, meme in enumerate(memes):
        try:
            eval_dict = evaluate_meme(paper_content, meme)
            f_score, c_score, e_score, avg_score = get_score(eval_dict)
            if f_score is None or c_score is None or e_score is None:
                f_score = f_score or 0.0
                c_score = c_score or 0.0
                e_score = e_score or 0.0
                avg_score = (f_score + c_score + e_score) / 3
            evaluated_memes.append({
                'meme': meme,
                'fidelity_score': f_score,
                'clarity_score': c_score,
                'engagement_score': e_score,
                'average_score': avg_score,
                'evaluation': eval_dict,
                'index': i
            })
        except Exception as e:
            print(f"Error evaluating meme {i}: {e}")
            evaluated_memes.append({
                'meme': meme,
                'fidelity_score': 0.0,
                'clarity_score': 0.0,
                'engagement_score': 0.0,
                'average_score': 0.0,
                'evaluation': {'error': str(e)},
                'index': i
            })
    evaluated_memes.sort(key=lambda x: x['average_score'], reverse=True)
    return evaluated_memes

# ------------------------------
# Main Exploration-Exploitation Loop
# ------------------------------

def exploration_exploitation_search(paper_content, max_iterations=6, k=5):
    """
    Returns:
      all_time_best_meme, history, all_time_best_score, iteration_scores (list of 6 dicts for iterations 1..6)
    """
    history = []
    iteration_scores = []  # <-- will capture scores for iterations 1..max_iterations
    all_time_best_score = -1
    all_time_best_meme = None

    print("=== INITIAL EXPLORATION ===")
    selection = select_meme_templates(paper_content, meme_templates, k=k)
    chosen_templates = selection['selected_templates']
    print(f"Selected templates: {chosen_templates}")
    print(f"Justification: {selection['justification']}")

    diverse_memes = generate_memes_from_templates(paper_content, chosen_templates)
    evaluated = evaluate_and_rank_memes(paper_content, diverse_memes)

    if not evaluated:
        print("Error: No memes were successfully generated or evaluated in initial exploration")
        # Create a placeholder so downstream logic doesn't break
        evaluated = [{
            'meme': 'N/A',
            'fidelity_score': 0.0,
            'clarity_score': 0.0,
            'engagement_score': 0.0,
            'average_score': 0.0,
            'evaluation': {'error': 'no candidates'},
            'index': 0
        }]

    best = evaluated[0]
    worst = evaluated[-1]
    all_time_best_meme = best['meme']
    all_time_best_score = best['average_score']

    history.append({
        "iteration": 0,
        "phase": "initial_exploration",
        "candidates": evaluated,
        "best_meme": best['meme'],
        "best_score": best['average_score'],
        "worst_meme": worst['meme'],
        "worst_score": worst['average_score'],
        "selected_templates": chosen_templates,
    })

    # Iterative contrastive improvement
    for t in range(1, max_iterations + 1):
        print(f"\n=== ITERATION {t} ===")
        selection = select_meme_templates(paper_content, meme_templates, k=k)
        chosen_templates = selection['selected_templates']
        print(f"Iteration {t} templates: {chosen_templates}")
        print(f"Reason: {selection['justification']}")

        contrastive_memes = generate_contrastive_memes(
            paper_content,
            history[-1]['best_meme'],
            history[-1]['worst_meme'],
            history[-1]['candidates'][0]['evaluation'],
            history[-1]['candidates'][-1]['evaluation'],
            chosen_templates
        )
        evaluated = evaluate_and_rank_memes(paper_content, contrastive_memes)

        if not evaluated:
            print(f"Warning: No memes generated in iteration {t}, using placeholder scores.")
            placeholder = {
                'meme': 'N/A',
                'fidelity_score': 0.0,
                'clarity_score': 0.0,
                'engagement_score': 0.0,
                'average_score': 0.0,
                'evaluation': {'error': f'no candidates in iteration {t}'},
                'index': 0
            }
            evaluated = [placeholder]
            best = placeholder
            worst = placeholder
        else:
            best = evaluated[0]
            worst = evaluated[-1]

        # Track all-time best
        if best['average_score'] > all_time_best_score:
            all_time_best_score = best['average_score']
            all_time_best_meme = best['meme']
            print(f"🚀 Improved to {all_time_best_score:.3f}")
        else:
            # Keep the all-time best meme/score for continuity in next iteration prompts
            best = {"meme": all_time_best_meme, "average_score": all_time_best_score}

        history.append({
            "iteration": t,
            "phase": "contrastive_exploration",
            "candidates": evaluated,
            "best_meme": best['meme'],
            "best_score": best['average_score'],
            "worst_meme": worst['meme'],
            "worst_score": worst['average_score'],
            "selected_templates": chosen_templates,
        })

        # Store per-iteration scores (based on the best candidate in this iteration)
        top = evaluated[0]
        iteration_scores.append({
            "iteration": t,
            "fidelity": float(top.get("fidelity_score", 0.0)),
            "clarity": float(top.get("clarity_score", 0.0)),
            "engagement": float(top.get("engagement_score", 0.0)),
            "average": float(top.get("average_score", 0.0)),
        })

    # Ensure exactly 6 iterations are recorded (pad if needed)
    # (This is defensive; normally max_iterations=6 guarantees this.)
    while len(iteration_scores) < 6:
        iteration_scores.append({
            "iteration": len(iteration_scores) + 1,
            "fidelity": 0.0,
            "clarity": 0.0,
            "engagement": 0.0,
            "average": 0.0
        })

    return all_time_best_meme, history, all_time_best_score, iteration_scores

# ------------------------------
# Main Driver
# ------------------------------
if __name__ == "__main__":
    all_paper_results = []
    final_scores = {'fidelity': [], 'clarity': [], 'engagement': [], 'average': []}

    for paper_id in os.listdir(config['data']['path'])[0:amount]:
        paper_path = os.path.join(config['data']['path'], paper_id)

        try:
            abstract, paper_text = read_json(paper_path)
            # Combine abstract and full text
            full_paper_text = f"Abstract:\n{abstract}\n\nFull Paper:\n{paper_text}"
        except Exception as e:
            print(f"Error reading paper {paper_id}: {e}")
            continue

        print(f"\n{'='*80}")
        print(f"PROCESSING PAPER: {paper_id}")
        print(f"{'='*80}")

        try:
            # STEP 1: Extract concise paper summary using analysis agents (called once)
            print("\n🔍 Analyzing paper to extract key ideas...")
            paper_summary = extract_paper_summary(full_paper_text)
            print(f"Paper Summary:\n{paper_summary}\n")

            # STEP 2: Use the summary for meme generation with 6 iterations
            final_meme, search_history, best_score, iteration_scores = exploration_exploitation_search(
                paper_summary, max_iterations=6
            )

            paper_results = {
                "paper_id": paper_id,
                "paper_summary": paper_summary,
                "final_best_meme": final_meme,
                "final_best_score": best_score,
                "iteration_scores": iteration_scores,   # <-- 6 iterations x 4 scores
                "search_history": search_history,
                "abstract": abstract
            }
            all_paper_results.append(paper_results)

            # Final detailed eval on the final best meme
            try:
                final_eval = evaluate_meme(paper_summary, final_meme)
                f_score, c_score, e_score, avg_score = get_score(final_eval)
            except Exception as e:
                print(f"Error in final evaluation: {e}")
                f_score = c_score = e_score = avg_score = 0.0

            final_scores['fidelity'].append(f_score or 0.0)
            final_scores['clarity'].append(c_score or 0.0)
            final_scores['engagement'].append(e_score or 0.0)
            final_scores['average'].append(avg_score or 0.0)

            print(f"\n📊 FINAL RESULTS for {paper_id}:")
            print(f"Best Score: {best_score:.3f}")
            print(f"Fidelity: {f_score:.3f}, Clarity: {c_score:.3f}, Engagement: {e_score:.3f}, Avg: {avg_score:.3f}")
            print(f"Best Meme Preview:\n{(final_meme or '')[:200]}...\n")

        except Exception as e:
            print(f"Error processing paper {paper_id}: {e}")
            continue

    # ------------------------------
    # Global Analysis
    # ------------------------------
    if all_paper_results:
        print(f"\n{'='*100}")
        print(f"OVERALL ANALYSIS ACROSS {len(all_paper_results)} PAPERS")
        print(f"{'='*100}")

        print(f"\n🎯 FINAL AVERAGE SCORES:")
        avg_fidelity = sum(final_scores['fidelity'])/len(final_scores['fidelity']) if final_scores['fidelity'] else 0.0
        avg_clarity = sum(final_scores['clarity'])/len(final_scores['clarity']) if final_scores['clarity'] else 0.0
        avg_engagement = sum(final_scores['engagement'])/len(final_scores['engagement']) if final_scores['engagement'] else 0.0
        avg_overall = sum(final_scores['average'])/len(final_scores['average']) if final_scores['average'] else 0.0

        print(f"  Fidelity: {avg_fidelity:.3f}")
        print(f"  Clarity: {avg_clarity:.3f}")
        print(f"  Engagement: {avg_engagement:.3f}")
        print(f"  Overall: {avg_overall:.3f}")

        output_data = {
            "summary": {
                "total_papers": len(all_paper_results),
                "average_scores": {
                    "fidelity": avg_fidelity,
                    "clarity": avg_clarity,
                    "engagement": avg_engagement,
                    "overall": avg_overall
                }
            },
            "detailed_results": all_paper_results
        }

        # Ensure output directory exists
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to {args.output}")
    else:
        print("No papers were successfully processed.")
