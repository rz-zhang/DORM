import json

def process_jsonl(input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            # Parse JSON line
            data = json.loads(line.strip())

            # Add -100 to the label list
            data['label'].append(-100.0)

            # Write the modified JSON line
            fout.write(json.dumps(data) + '\n')

# Example usage
input_file = './data/hs2/val_reg.jsonl'
output_file = './data/hs2/val_reg_10d_label.jsonl'
process_jsonl(input_file, output_file)