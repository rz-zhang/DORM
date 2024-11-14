import argparse
import json

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file", type=str, default="data/magpie/Magpie-Pro-DPO-100K-v0.1-train-original.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output-file", type=str, default="data/magpie/Magpie-Pro-DPO-100K-v0.1-train-scaled.jsonl",
        help="Path to output JSONL file"
    )
    return parser.parse_args()

def get_reward_bounds(input_file):
    """Get global min and max rewards from the entire file."""
    min_reward = float('inf')
    max_reward = float('-inf')

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            reward = data['label'][-1]  # Get the last value in label list
            if reward != -100.0:  # Skip padding values
                min_reward = min(min_reward, reward)
                max_reward = max(max_reward, reward)

    return min_reward, max_reward

def scale_reward(reward, min_reward, max_reward):
    """Scale reward to range [0,5]."""
    if max_reward == min_reward:
        return 2.5  # Middle of range if min=max
    return 5.0 * (reward - min_reward) / (max_reward - min_reward)

def process_file(input_file, output_file, min_reward, max_reward):
    """Process the file and scale rewards."""
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            data = json.loads(line)
            label = data['label']

            # Scale only the last value if it's not a padding value
            if label[-1] != -100.0:
                label[-1] = scale_reward(label[-1], min_reward, max_reward)

            # Write the modified data
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

def main(args):
    # First pass: get min and max rewards
    print("Finding global reward bounds...")
    min_reward, max_reward = get_reward_bounds(args.input_file)
    print(f"Reward bounds - Min: {min_reward:.3f}, Max: {max_reward:.3f}")

    # Second pass: scale rewards and write output
    print("Scaling rewards...")
    process_file(args.input_file, args.output_file, min_reward, max_reward)
    print(f"Successfully processed rewards and saved to {args.output_file}")

if __name__ == "__main__":
    main(prepare_args())