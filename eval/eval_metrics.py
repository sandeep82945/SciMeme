from transformers import pipeline, AutoModel, AutoTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import yaml
import json

class BaseMetricEvaluator:
    """
    A base class to handle the common logic for calling the LLM and parsing its response.
    Each specific metric will inherit from this class.
    """
    def __init__(self, client=None, model_name: str = "gpt-4.1", temperature: float = 0.2):
        if client is None:
            raise Exception("OpenAI client is not available.")
        self.model = model_name
        self.temperature = temperature
        self.system_prompt = ""
        self.client = client

    def _create_prompt_messages(self, paper_content: str, meme_text: str) -> list:
        """Creates the message structure for the API call."""
        user_content = f"""
        ### Research Paper Content
        {paper_content}

        ### Generated Meme Text
        {meme_text}
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

    def evaluate(self, paper_content: str, meme_text: str) -> dict:
        """
        Calls the LLM with the provided content and returns the structured evaluation.
        """
        messages = self._create_prompt_messages(paper_content, meme_text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content
            return json.loads(response_content)
        except Exception as e:
            print(f"An error occurred during API call for {self.__class__.__name__}: {e}")
            return {"score": None, "reasoning": "Error during evaluation."}


class FidelityScoreEvaluator(BaseMetricEvaluator):
    """Evaluates scientific fidelity using LLM as judge approach."""
    def __init__(self, client=None, model_name: str = "gpt-4.1", temperature: float = 0.2):
        super().__init__(client, model_name, temperature)

        self.system_prompt = """
    You are an expert scientific reviewer. Your only task is to evaluate *scientific fidelity* of a meme generated from a research paper.

    ### Definition
    Scientific fidelity = the degree to which the meme faithfully preserves the research paper's **core claim or contribution** without distortion.  
    It does not measure humor, creativity, clarity, or engagement — only whether the scientific meaning remains accurate.

    ### Evaluation Steps
    1. **Extract Core Claim:** Identify the central scientific contribution or innovation in the research paper.  
    2. **Extract Meme Claim:** Identify the main scientific claim communicated by the meme caption.  
    3. **Compare:** Judge how closely the meme's claim aligns with the paper's true claim, checking for:
    - Factual correctness (no errors or contradictions)  
    - Preservation of scope/limitations (not exaggerated beyond what the paper claims)  
    - Integrity of meaning (simplification is acceptable if truth is preserved)  

    ### Scoring Rubric (1–5)
    - **5 (Exact Fidelity):** Meme precisely captures the paper's core contribution. Simplification is valid and retains all essential truth.  
    - **4 (High Fidelity):** Meme captures the main claim correctly but omits or blurs a nuance (e.g., scope, limitation, or significance).  
    - **3 (Partial Fidelity):** Meme conveys the general topic but misses or alters the specific contribution. The audience may get the field but not the research advance.  
    - **2 (Low Fidelity):** Meme introduces a major distortion, exaggeration, or misleading framing of the research.  
    - **1 (No Fidelity):** Meme is scientifically inaccurate, contradictory, or unrelated to the paper's claim.  

    ### Output Format
    Return your evaluation in JSON format with the following structure:

    {
    "score": [1–5],
    "reasoning": "Brief explanation of why this score was assigned, citing paper vs meme claims.",
    "correct_claim_summary": "One-sentence summary of the paper's true core contribution.",
    "meme_claim_summary": "One-sentence summary of what the meme conveys.",
    "scope_check": "Does the meme preserve or exaggerate the scope/limitations of the research?",
    "factual_status": "Accurate | Partially Accurate | Inaccurate",
    "key_misrepresentations": ["List of distortions, exaggerations, or missing nuances, if any."]
    }

    """

    def evaluate(self, paper_content: str, meme_text: str) -> dict:
        """Evaluate scientific fidelity using LLM as judge approach.
        Always return a score, but keep other fields if available.
        """
        messages = self._create_prompt_messages(paper_content, meme_text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)

            # Extract and validate score
            score = result.get("score", None)
            if not isinstance(score, int) or score < 1 or score > 5:
                score = 3
                result["score"] = score  # overwrite with safe fallback

            return result  # keep score + any other fields

        except Exception as e:
            print(f"An error occurred during API call for FidelityScoreEvaluator: {e}")
            # Return minimal dict with safe defaults
            return {
                "score": 3,
                "reasoning": f"Error during evaluation: {e}"
            }

class ClarityInterpretabilityEvaluator:
    """
    Evaluates meme clarity using a detailed, persona-driven prompt and the FRI score rubric.
    """
    def __init__(self, client=None, model_name: str = "gpt-4.1", temperature: float = 0.5):
        if client is None:
            raise Exception("OpenAI client is not available.")
        self.model = model_name
        self.temperature = temperature
        self.client = client

    def evaluate(self, research_idea: str, meme_caption: str, meme_template_description: str) -> dict:
        prompt_template = f"""
        # CONTEXT
        You are an AI assistant tasked with evaluating the clarity and interpretability of a meme designed to communicate a novel scientific idea. 
        The target audience is graduate students and early-career researchers in computer science (especially NLP/ML). 
        Your goal is to determine how well the meme conveys the core message to this audience.

        # THE MEME TO EVALUATE
        - Meme Template: "{meme_template_description}"
        - Meme Caption: "{meme_caption}"

        # THE GROUND TRUTH (Core Scientific Message)
        This is the idea the meme is *supposed* to communicate. Do NOT use this information for your initial interpretation.
        - Core Message: "{research_idea}"

        # YOUR TASK
        Follow these steps precisely:
        1. **Initial Interpretation:** Based ONLY on the meme template and caption, explain what you think the meme is about. What is the main point or joke? Be honest about any confusion.
        2. **Comparison:** Now, read the "GROUND TRUTH (Core Scientific Message)" provided above.
        3. **Scoring:** Compare your initial interpretation with the ground truth. Using the rubric below, assign a single "FRI Score" from 1 to 3 that reflects how well the meme communicated the core message to the target audience.
        4. **Justification:** Briefly explain why you gave that score. Point out what was clear, what was confusing, or what led to your interpretation.

        # SCORING RUBRIC (for Graduate/PhD-level Researchers in CS/NLP)
        - **FRI Score 3 (Clear):** The meme effectively conveyed the core scientific message. A graduate student or early-career researcher in the field would immediately understand the idea without extra explanation.
        - **FRI Score 2 (Partially Clear):** The meme conveyed part of the message but left out key details or nuance. A graduate student or early-career researcher would get the gist but might misinterpret some aspects or need clarification.
        - **FRI Score 1 (Unclear):** The meme failed to convey the intended message. Even a graduate student or early-career researcher in the field would likely misunderstand or remain confused.

        # OUTPUT FORMAT
        Provide your response in JSON format with the following keys:
        - "fri_score": A single number from 1 to 3
        - "justification": Your detailed justification from step 4
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant simulating a graduate student in computer science evaluating a scientific meme. Output your final answer in JSON format."},
                    {"role": "user", "content": prompt_template}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"fri_score": None, "justification": f"An error occurred during API call: {e}"}


class EngagementPotentialEvaluator(BaseMetricEvaluator):
    """Evaluates the meme's potential to be shared and discussed."""
    def __init__(self, client=None, model_name: str = "gpt-4.1", temperature: float = 0.2):
        super().__init__(client, model_name, temperature)
        self.system_prompt = """
        You are an expert scientific reviewer.
        You will assign ONE overall Engagement Potential score (1–5) to a scientific meme.

        DEFINITION (what to judge)
        - Predicted resonance and shareability within the intended scientific audience, driven by:
        (a) Humor strength, (b) Template synergy with the joke (c) Offensiveness.

        ORTHOGONALITY (what NOT to judge)
        - Do NOT judge scientific fidelity/factuality of the meme.
        - Do NOT judge explanatory clarity/interpretability.
        - Ignore technical depth, pedagogy, or didactic quality of research problem.

        HARD RULE (binary, overrides everything)
        - If the meme can reasonably be read as offensive/disparaging toward previous work, specific papers, authors, labs, or reviewers,
        set the FINAL SCORE to 1 (e.g., insults, ridicule like "trash", "idiots", demeaning comparisons).
        - Also set 1 for any slurs/hate toward a protected group.
        - When in doubt about this rule, choose 1.

        SCORING GUIDE (1–5) — apply ONLY if the hard rule did not trigger
        5: Exceptional resonance: "Banger" for the target community; strong in-joke/twist; template amplifies the gag; highly shareable (e.g., on academic Twitter/X, in lab Slacks).
        4: Highly Engaging: Strong resonance. Clearly resonant; funny and relatable; minor staleness or slight niche limits.
        3: Moderately resonant; mild chuckle or niche appeal; shareable within a small sub-group.
        2: Low resonance; stale/forced joke or weak template synergy; limited sharing expected.
        1: Doesn't land (confusing/unfunny) OR assigned 1 by the hard rule.

        ASSUMPTIONS
        - If audience unspecified, assume STEM/ML graduate-level community.
        - Treat any visual layout as the "template" if no named template is given.

        OUTPUT
        Return your evaluation in JSON format with the following structure:
        {"score": <int 1-5>, "reasoning": "<≤20 words; focus on resonance/shareability; if rule triggered, say 'offensive to previous work'>"}
        """


class Scores:
    def __init__(self, client=None, config=None):
        if config is None:
            raise Exception("Configuration is required")
        
        # Get model configurations
        model_config = config.get("models", {})
        self.model_name = model_config.get("default", "gpt-4.1")
        
        # Get temperature settings
        temp_config = config.get("temperatures", {})
        
        # Initialize evaluators with config parameters
        self.fidelity = FidelityScoreEvaluator(
            client=client, 
            model_name=self.model_name,
            temperature=temp_config.get("fidelity", 0.0)
        )
        self.engagement = EngagementPotentialEvaluator(
            client=client, 
            model_name=self.model_name,
            temperature=temp_config.get("engagement", 0.0)
        )
        self.clarity = ClarityInterpretabilityEvaluator(
            client=client, 
            model_name=self.model_name,
            temperature=temp_config.get("clarity", 0.0)
        )

    def __call__(self, abstract=None, meme=None, meme_template=None):
        fidelity_result = self.fidelity.evaluate(abstract, meme)
        engagement_result = self.engagement.evaluate(abstract, meme)
        clarity_result = self.clarity.evaluate(
            research_idea=abstract,
            meme_caption=meme,
            meme_template_description=meme_template
        )
        
        combined_result = {
            "fidelity": fidelity_result,
            "engagement": engagement_result,
            "clarity": clarity_result
        }
        return combined_result


def main():
    # Load configuration
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Get API key from config
    openai_key = config.get("api_keys", {}).get("openai")
    
    if not openai_key:
        print("Error: OpenAI API key not found in config.yaml")
        return

    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=openai_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return

    # Get test parameters from config
    test_config = config.get("test_parameters", {})
    abstract = test_config.get("abstract", "The novel idea of Attention Is All You Need...")
    output_file = test_config.get("output_file", "../results/eval_results.json")
    
    # Initialize scoring system
    score_system = Scores(client=client, config=config)
    
    # Test cases (can also be moved to config if needed)
    tests = [
        {"Distracted Boyfriend": "Machine Learning Community, RNNs and CNNs, Attention Is All You Need"},
        {"Drake Hotline Bling": "Using recurrence for sequence modeling, Using self-attention for everything"},
        # Add more test cases as needed
    ]

    results = {}
    for test in tests:
        for meme_template, meme_caption in test.items():
            print(f"Evaluating meme template: {meme_template}")
            eval_result = score_system(abstract, meme_caption, meme_template)
            results[f"{meme_template}: {meme_caption[:50]}..."] = eval_result
            print(results)
            break  # Remove exit(0) and use break instead for cleaner exit
            
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()