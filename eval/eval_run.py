from eval.eval_metrics import Scores
import yaml
from openai import OpenAI



import argparse
import re
from datetime import datetime

from openai import OpenAI

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

openai_key = config['api_keys']['openai']
openai_client = OpenAI(api_key=openai_key)

score_system = Scores(client=openai_client, config=config)

def evaluate_meme(abstract, meme_text):
    """
    Evaluate a meme using the scoring system.
    """
    try:
        # Call the scoring system
        eval_results = score_system(
            abstract=abstract,
            meme=meme_text,
        )
        
        return eval_results
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {
            'fidelity': {'score': None, 'reasoning': f'Evaluation error: {e}'},
            'engagement': {'score': None, 'reasoning': f'Evaluation error: {e}'},
            'clarity': {'fri_score': None, 'justification': f'Evaluation error: {e}'}
        }

def get_score(dic):
    return dic['fidelity']['score'], dic['clarity']['fri_score'],dic['engagement']['score'], dic['fidelity']['score'] + dic['engagement']['score'] + dic['clarity']['fri_score']