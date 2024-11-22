import jsonlines
import numpy as np
import collections
from tqdm import tqdm
import os
from bayes_opt import BayesianOptimization
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import EXAMPLE_COUNTS, SUBSET_MAPPING
import argparse

# Define bounds for all attributes
PARAM_BOUNDS = {
    "quality": (0.0, 2.0),
    "toxicity": (-1.0, 1.0),
    "humor": (-1.0, 1.0),
    "creativity": (-1.0, 1.0),
    "helpfulness": (0.0, 2.0),
    "correctness": (0.0, 1.0),
    "coherence": (0.0, 1.0),
    "complexity": (0.0, 1.0),
    "verbosity": (-1.0, 1.0),
    "magpie_score": (0.0, 3.0),
    "offset_bias": (0.0, 2.0),
    "wild_guard": (0.0, 2.0)
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
    "verbosity": -0.6,
    "magpie_score": 0.5,
    "offset_bias": 0.5,
    "wild_guard": 0.5
}

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True,
                       help="Path to the prediction file")
    parser.add_argument("--n_trials", type=int, default=10,
                       help="Number of parallel optimization trials")
    parser.add_argument("--n_iter", type=int, default=100,
                       help="Number of iterations per trial")
    return parser.parse_args()

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

class WeightOptimizer:
    def __init__(self, pred_file):
        self.pred_file = pred_file
        self.best_weights = None
        self.best_accuracy = 0
        self.best_section_accuracies = {}

    def objective_function(self, **kwargs):
        """Objective function for Bayesian optimization"""
        weights = {k: float(v) for k, v in kwargs.items()}
        accuracy, section_accuracies = calculate_accuracy_and_sections(self.pred_file, weights)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_weights = weights.copy()
            self.best_section_accuracies = section_accuracies
            print(f"\nNew best accuracy: {accuracy:.4f}")
            print(f"Weights: {weights}")

        return accuracy

    def optimize(self, n_iter=100):
        # Initialize optimizer
        optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=PARAM_BOUNDS,
            random_state=42
        )

        # Load previous weights as initial point
        optimizer.probe(
            params={k: previous_weights[k] for k in PARAM_BOUNDS.keys()},
            lazy=True
        )

        # Run optimization
        optimizer.maximize(
            init_points=10,
            n_iter=n_iter,
        )

        return self.best_weights, self.best_accuracy, self.best_section_accuracies

def parallel_search(pred_file, n_trials=5):
    """Run multiple optimization instances in parallel"""
    best_results = []

    def run_optimization():
        optimizer = WeightOptimizer(pred_file)
        return optimizer.optimize(n_iter=50)  # Fewer iterations per trial

    with ThreadPoolExecutor(max_workers=min(n_trials, os.cpu_count())) as executor:
        futures = [executor.submit(run_optimization) for _ in range(n_trials)]

        for future in tqdm(as_completed(futures), total=n_trials, desc="Running parallel trials"):
            best_results.append(future.result())

    # Return the best result across all trials
    return max(best_results, key=lambda x: x[1])

if __name__ == "__main__":
    args = prepare_args()

    print("Evaluating previous weights...")
    prev_accuracy, prev_section_accuracies = calculate_accuracy_and_sections(args.pred_file, previous_weights)
    print(f"Previous Weights Overall Accuracy: {prev_accuracy:.4f}")
    for section, acc in prev_section_accuracies.items():
        print(f"Previous Weights {section} Accuracy: {acc:.4f}")

    print("\nStarting optimization...")
    best_weights, best_accuracy, best_section_accuracies = parallel_search(args.pred_file, n_trials=args.n_trials)

    print("\nFinal Results:")
    print(f"Best Weights: {best_weights}")
    print(f"Best Overall Accuracy: {best_accuracy:.4f}")
    for section, acc in best_section_accuracies.items():
        print(f"Best {section} Accuracy: {acc:.4f}")