import json
import os
import argparse
from pathlib import Path

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output JSONL files"
    )
    parser.add_argument(
        "--label-length",
        type=int,
        required=True,
        help="Target length for label list"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_processed",
        help="Suffix to add to processed files"
    )
    return parser.parse_args()

def process_jsonl(input_file, output_file, target_length):
    """Process a single JSONL file, padding labels to target length."""
    print(f"Processing {input_file}...")
    count = 0

    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            try:
                # Parse JSON line
                data = json.loads(line.strip())

                # Get current label length
                current_length = len(data['label'])

                # Pad with -100.0 if needed
                if current_length < target_length:
                    padding_needed = target_length - current_length
                    data['label'].extend([-100.0] * padding_needed)
                elif current_length > target_length:
                    print(f"Warning: Found label of length {current_length} > {target_length} in {input_file}")
                    data['label'] = data['label'][:target_length]

                # Write the modified JSON line
                fout.write(json.dumps(data) + '\n')
                count += 1

            except json.JSONDecodeError as e:
                print(f"Error processing line in {input_file}: {e}")
                continue

    print(f"Processed {count} lines in {input_file}")
    return count

def main():
    args = prepare_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all JSONL files in input directory
    total_files = 0
    total_lines = 0

    for input_file in Path(args.input_dir).glob('*.jsonl'):
        # Construct output filename
        base_name = input_file.stem
        output_name = f"{base_name}{args.suffix}.jsonl"
        output_file = Path(args.output_dir) / output_name

        # Process file
        lines_processed = process_jsonl(input_file, output_file, args.label_length)
        total_files += 1
        total_lines += lines_processed

    print(f"\nSummary:")
    print(f"Files processed: {total_files}")
    print(f"Total lines processed: {total_lines}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()