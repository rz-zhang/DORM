import json

def select_subset(input_file, output_file, num_entries):
 """
 Selects a specified number of entries from a JSON Lines file and writes them to a new file.

 Args:
     input_file (str): The path to the source JSON Lines file.
     output_file (str): The path where the subset will be saved.
     num_entries (int): The number of entries to select.
 """
 try:
     with open(input_file, 'r', encoding='utf-8') as infile, \
          open(output_file, 'w', encoding='utf-8') as outfile:
         # Initialize a counter to track the number of entries processed
         count = 0
         for line in infile:
             if count < num_entries:
                 outfile.write(line)
                 count += 1
             else:
                 break
 except FileNotFoundError:
     print(f"Error: The file {input_file} does not exist.")
 except Exception as e:
     print(f"An error occurred: {e}")

# Example usage:
select_subset('data/magpie/Magpie-Pro-DPO-100K-v0.1-train-scaled.jsonl', 'data/magpie/Magpie-Pro-DPO-80K-v0.1-train-scaled.jsonl', 80000)
