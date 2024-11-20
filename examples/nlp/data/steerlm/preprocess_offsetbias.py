import argparse
import json
import random
from datasets import load_dataset
from common import (
    ASSISTANT_TURN_TEMPLATE,
    LABEL_PREFIX,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    USER_TURN_TEMPLATE,
)

NUM_LABEL_PLACEHOLDER = 10
POS_LABEL = 4.0
NEG_LABEL = 1.0

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/offsetbias/offestbias-train-16k.jsonl",
        help="Path to training data output JSONL file"
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default="data/offsetbias/offestbias-test-1k.jsonl",
        help="Path to test data output JSONL file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting"
    )
    return parser.parse_args()

def format_conversation(instruction: str, response: str) -> str:
    """Format a conversation following the system prompt and turn templates."""
    text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)
    text += USER_TURN_TEMPLATE.format(value=instruction)
    text += ASSISTANT_TURN_TEMPLATE.format(value=response)
    text += LABEL_PREFIX
    return text

def process_sample(item, fout):
    """Process a single sample from the offsetbias dataset."""
    instruction = item['instruction']

    # Process output_1
    text_1 = format_conversation(instruction, item['output_1'])
    score_1 = POS_LABEL if item['label'] == 1 else NEG_LABEL
    newline_1 = {
        "text": text_1,
        "label": [-100.0] * NUM_LABEL_PLACEHOLDER + [score_1]
    }
    fout.write(json.dumps(newline_1, ensure_ascii=False) + "\n")

    # Process output_2
    text_2 = format_conversation(instruction, item['output_2'])
    score_2 = POS_LABEL if item['label'] == 2 else NEG_LABEL
    newline_2 = {
        "text": text_2,
        "label": [-100.0] * NUM_LABEL_PLACEHOLDER + [score_2]
    }
    fout.write(json.dumps(newline_2, ensure_ascii=False) + "\n")

def main(args):
    # Set random seed
    random.seed(args.seed)

    # Load dataset
    try:
        dataset = load_dataset("NCSOFT/offsetbias")['train']  # Only train split exists
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Convert to list and shuffle
    all_samples = list(dataset)
    random.shuffle(all_samples)

    # Split into train (8k) and test (500) samples
    # This will result in 16k and 1k examples respectively after processing
    train_samples = all_samples[:8000]
    test_samples = all_samples[8000:8500]

    # Process training data
    print(f"Processing {len(train_samples)} training samples...")
    with open(args.train_output, "w", encoding="utf-8") as fout:
        for item in train_samples:
            process_sample(item, fout)

    # Process test data
    print(f"Processing {len(test_samples)} test samples...")
    with open(args.test_output, "w", encoding="utf-8") as fout:
        for item in test_samples:
            process_sample(item, fout)

    print(f"Successfully processed dataset:")
    print(f"Training data saved to: {args.train_output} ({len(train_samples) * 2} examples)")
    print(f"Test data saved to: {args.test_output} ({len(test_samples) * 2} examples)")

if __name__ == "__main__":
    main(prepare_args())