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

NUM_LABEL_PLACEHOLDER = 11
POS_LABEL = 4.0
NEG_LABEL = 1.0

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/wildguard/wildguard-train-12k.jsonl",
        help="Path to training data output JSONL file"
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default="data/wildguard/wildguard-test-1k.jsonl",
        help="Path to test data output JSONL file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting"
    )
    return parser.parse_args()

def format_conversation(messages: list) -> str:
    """Format a conversation following the system prompt and turn templates."""
    text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)

    # Process each message in the conversation
    for msg in messages:
        if msg["role"] == "user":
            text += USER_TURN_TEMPLATE.format(value=msg["content"])
        elif msg["role"] == "assistant":
            text += ASSISTANT_TURN_TEMPLATE.format(value=msg["content"])

    text += LABEL_PREFIX
    return text

def process_sample(item, fout):
    """Process a single sample from the wildguard dataset."""
    # Process chosen response
    text_chosen = format_conversation(item['chosen'])
    newline_chosen = {
        "text": text_chosen,
        "label": [-100.0] * NUM_LABEL_PLACEHOLDER + [POS_LABEL]
    }
    fout.write(json.dumps(newline_chosen, ensure_ascii=False) + "\n")

    # Process rejected response
    text_rejected = format_conversation(item['rejected'])
    newline_rejected = {
        "text": text_rejected,
        "label": [-100.0] * NUM_LABEL_PLACEHOLDER + [NEG_LABEL]
    }
    fout.write(json.dumps(newline_rejected, ensure_ascii=False) + "\n")

def main(args):
    # Set random seed
    random.seed(args.seed)

    # Load dataset
    try:
        dataset = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")['train']
        # Filter for wildguard source
        dataset = [item for item in dataset if item['source'] == 'wildguard']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Convert to list and shuffle
    all_samples = list(dataset)
    random.shuffle(all_samples)

    # Calculate split sizes (assuming similar 16:1 ratio as offsetbias)
    total_size = len(all_samples)
    train_size = 6000  # Keep similar ratio

    # Split into train and test samples
    train_samples = all_samples[:train_size]
    test_samples = all_samples[train_size:train_size+500]

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

    total_train_examples = len(train_samples) * 2  # Each sample produces 2 examples
    total_test_examples = len(test_samples) * 2

    print(f"Successfully processed dataset:")
    print(f"Training data saved to: {args.train_output} ({total_train_examples} examples)")
    print(f"Test data saved to: {args.test_output} ({total_test_examples} examples)")

if __name__ == "__main__":
    main(prepare_args())