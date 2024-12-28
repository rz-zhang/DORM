import json
import os

file_sources = {
    'train/processed/hs2-train-reg-20k_processed.jsonl': 'hs2',
    'train/processed/oasst2-train-reg-20k_processed.jsonl': 'oasst2',
    'train/processed/wildguard-train-12k_processed.jsonl': 'wildguard',
    'train/processed/Magpie-Pro-DPO-v0.1-train-scaled-12k_processed.jsonl': 'magpie',
    'train/processed/offsetbias-train-16k_processed.jsonl': 'offsetbias'
}

combined_data = []

for file_name, source in file_sources.items():
    with open(file_name, 'r') as f:
        for line in f:
            data = json.loads(line)
            data['source'] = source
            combined_data.append(data)

with open('dorm-train-80k-with-source.jsonl', 'w') as outfile:
    for entry in combined_data:
        json.dump(entry, outfile)
        outfile.write('\n')