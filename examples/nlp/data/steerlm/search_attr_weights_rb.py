import jsonlines
import numpy as np
import collections
from itertools import product
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import EXAMPLE_COUNTS, SUBSET_MAPPING

helpfulness_weights = np.arange(0.5, 1.1, 0.1)
correctness_weights = np.arange(0.4, 1.1, 0.1)
coherence_weights = np.arange(0, 1.1, 0.1)
complexity_weights = np.arange(0, 1.1, 0.1)
verbosity_weights = np.arange(-0.5, 0.5, 0.1)

fixed_weights = {
    "quality": 0.7,
    "toxicity": -0.8,
    "humor": 0.2,
    "creativity": 0.4,
}

previous_weights = {
    "quality": 0.6,
    "toxicity": 0.8,
    "humor": -0.2,
    "creativity": -0.1,
    "helpfulness": 0.8,
    "correctness": 0.6,
    "coherence": 0.15,
    "complexity": 0.6,
    "verbosity": -0.6
}

# previous_weights = {
#     "quality": 0,
#     "toxicity": 0,
#     "humor": 0,
#     "creativity": 0,
#     "helpfulness": 0.8,
#     "correctness": 0.6,
#     "coherence": 0.15,
#     "complexity": 0.6,
#     "verbosity": -0.6
# }

def calculate_score(label, weights):
    parts = label.split(',')
    score = 0
    for part in parts:
        attribute, value = part.split(':')
        value = float(value)
        weight = weights.get(attribute.strip(), 0)
        score += value * weight
    return score

def calculate_accuracy_and_sections(pred_file, attribute_weights):
    sample_id_to_conversations = collections.defaultdict(list)
    results = []

    with jsonlines.open(pred_file) as reader:
        for obj in reader:
            sample_id_to_conversations[obj["sample_id"]].append(obj)

    subset_results = collections.defaultdict(lambda: {'correct': 0, 'total': 0})

    for sample_id, conversations in sample_id_to_conversations.items():
        if len(conversations) != 2:
            continue
        if conversations[0]["prefer"] == 1:
            chosen = conversations[0]
            rejected = conversations[1]
        else:
            chosen = conversations[1]
            rejected = conversations[0]

        chosen_score = calculate_score(chosen["conversations"][1]["label"], attribute_weights)
        rejected_score = calculate_score(rejected["conversations"][1]["label"], attribute_weights)

        result = chosen_score > rejected_score
        results.append(result)

        subset = chosen["subset"]
        subset_results[subset]['correct'] += int(result)
        subset_results[subset]['total'] += 1

    section_accuracies = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, subset_results)
    overall_accuracy = sum(section_accuracies.values()) / len(section_accuracies.keys())

    return overall_accuracy, section_accuracies

def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                correct = metrics[test]['correct']
                total = metrics[test]['total']
                accuracy = correct / total if total > 0 else 0
                total_weighted_score += accuracy * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores

def search_best_weights(pred_file):
    best_weights = None
    best_accuracy = 0
    best_section_accuracies = {}

    print("Evaluating previous weights...")
    previous_accuracy, previous_section_accuracies = calculate_accuracy_and_sections(pred_file, previous_weights)
    print(f"Previous Weights Overall Accuracy: {previous_accuracy:.4f}")
    for section, acc in previous_section_accuracies.items():
        print(f"Previous Weights {section} Accuracy: {acc:.4f}")

    best_weights = previous_weights
    best_accuracy = previous_accuracy
    best_section_accuracies = previous_section_accuracies

    total_combinations = len(helpfulness_weights) * len(correctness_weights) * len(coherence_weights) * len(complexity_weights) * len(verbosity_weights)
    combinations = list(product(helpfulness_weights, correctness_weights, coherence_weights, complexity_weights, verbosity_weights))

    def evaluate_combination(combination):
        helpfulness, correctness, coherence, complexity, verbosity = combination
        attribute_weights = {
            **fixed_weights,
            "helpfulness": helpfulness,
            "correctness": correctness,
            "coherence": coherence,
            "complexity": complexity,
            "verbosity": verbosity
        }
        accuracy, section_accuracies = calculate_accuracy_and_sections(pred_file, attribute_weights)
        return accuracy, section_accuracies, attribute_weights

    # 确定最佳的线程池大小
    optimal_thread_count = min(32, os.cpu_count() + 4)
    with ThreadPoolExecutor(max_workers=optimal_thread_count) as executor:
        futures = [executor.submit(evaluate_combination, combination) for combination in combinations]

        for future in tqdm(as_completed(futures), total=total_combinations, desc="Searching weights", leave=True):
            accuracy, section_accuracies, attribute_weights = future.result()

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = attribute_weights
                best_section_accuracies = section_accuracies
                print(f"Best Weights: {best_weights}")
                print(f"Best Overall Accuracy: {best_accuracy:.4f}")
                for section, acc in best_section_accuracies.items():
                    print(f"Best {section} Accuracy: {acc:.4f}")

    return best_weights, best_accuracy, best_section_accuracies

if __name__ == "__main__":
    import os
    pred_file = "data/rewardbench/1112_pred_12b_oh_40k_lr3e6_3epoch_1node.jsonl"
    best_weights, best_accuracy, best_section_accuracies = search_best_weights(pred_file)
    print(f"Best Weights: {best_weights}")
    print(f"Best Overall Accuracy: {best_accuracy:.4f}")
    for section, acc in best_section_accuracies.items():
        print(f"Best {section} Accuracy: {acc:.4f}")
