import argparse
import json
from datasets import load_dataset

from common import (
    ALL_STEERLM_ATTRIBUTES,
    ASSISTANT_TURN_TEMPLATE,
    LABEL_PREFIX,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    USER_TURN_TEMPLATE,
)

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-file", type=str, required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to process"
    )
    return parser.parse_args()

def format_conversation(instruction, response):
    """Format a conversation following the system prompt and turn templates."""
    text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)
    text += USER_TURN_TEMPLATE.format(value=instruction)
    text += ASSISTANT_TURN_TEMPLATE.format(value=response)
    text += LABEL_PREFIX
    return text

def process_sample(item, fout):
    """Process a single sample from the dataset."""
    # Get basic information
    instruction = item.get('instruction', '')
    rewards = item.get('rewards_armorm', [])
    responses = item.get('responses', [])

    if not rewards or not responses:
        return

    scores = [r['score'] if isinstance(r, dict) and 'score' in r else r for r in rewards]

    # Find indices for chosen and rejected responses
    chosen_idx = max(range(len(scores)), key=lambda i: scores[i])
    rejected_idx = min(range(len(scores)), key=lambda i: scores[i])

    # Process chosen response
    if chosen_idx < len(responses):
        chosen_response = responses[chosen_idx]
        text = format_conversation(instruction, chosen_response)
        # First four values are -100.0 as in example
        newline = {
            "text": text,
            "label": [-100.0] * 9 + [scores[chosen_idx]]
        }
        fout.write(json.dumps(newline, ensure_ascii=False) + "\n")

    # Process rejected response
    if rejected_idx < len(responses):
        rejected_response = responses[rejected_idx]
        text = format_conversation(instruction, rejected_response)
        newline = {
            "text": text,
            "label": [-100.0] * 9 + [scores[rejected_idx]]
        }
        fout.write(json.dumps(newline, ensure_ascii=False) + "\n")

def main(args):
    # Load dataset
    try:
        dataset = load_dataset("Magpie-Align/Magpie-Pro-DPO-100K-v0.1")[args.split]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Process samples
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for item in dataset:
            process_sample(item, fout)

    print(f"Successfully processed dataset and saved to {args.output_file}")

if __name__ == "__main__":
    main(prepare_args())