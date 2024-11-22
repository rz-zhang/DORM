import argparse
import collections
import json
import os
import datetime
from typing import List

import jsonlines
import numpy as np
import socket
from common import (
    ALL_STEERLM_ATTRIBUTES,
    ASSISTANT_TURN_TEMPLATE,
    LABEL_PREFIX,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    USER_TURN_TEMPLATE,
    EXAMPLE_COUNTS,
    SUBSET_MAPPING,
)
from pytriton.client import FuturesModelClient
from tqdm import tqdm, trange
import sys

ALL_STEERLM_ATTRIBUTES = ALL_STEERLM_ATTRIBUTES + ["magpie_score"]


def test_server_connection(host, port, message="hello world!"):
    try:
        with socket.create_connection((host, port)) as sock:
            sock.sendall(message.encode('utf-8'))
            response = sock.recv(1024)
            print(f"Received: {response.decode('utf-8')}")
    except socket.error as e:
        print(f"Unable to connect to {host}:{port}. Error: {e}")

def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                # Ensure metrics[test]['correct'] and metrics[test]['total'] are available
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

def _str_list2numpy(str_list: List[str]) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_attr", type=int, default=9)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--model_name", type=str, default="reward_model")
    parser.add_argument("--add-eos", action="store_true")
    parser.add_argument("--pref_sets", action="store_true", help="Flag to indicate preference sets evaluation")
    return parser.parse_args()


def get_reward(
    sentences: List[str], host="localhost", port=5555, model_name="reward_model",
):
    sentences = _str_list2numpy(sentences)

    futures = []

    with FuturesModelClient(f"{host}:{port}", model_name) as client:
        for sen in np.split(sentences, sentences.shape[0]):
            future = client.infer_batch(sentences=sen)
            futures.append(future)

    all_result_dicts = [f.result() for f in futures]
#     print(output_dict.keys())

    all_rewards, all_exceeded = [], []
    all_context_tokens, all_hidden_states = [], []

    for output_dict in all_result_dicts:
        reward_out = output_dict["rewards"].flatten().tolist()

        all_rewards.append(reward_out)
#         all_exceeded += output_dict["exceeded"].tolist()
#         context_tokens = output_dict["context_tokens"]
#         if isinstance(context_tokens, np.ndarray):
#             context_tokens = context_tokens.tolist()
#         elif torch.is_tensor(context_tokens):
#             context_tokens = context_tokens.cpu().numpy().tolist()
#         all_context_tokens.append(context_tokens)

    return all_rewards


def get_key(l):
    convs = [c["value"] for c in l["conversations"]]
    return "".join(convs)


def calculate_score(label, weights):
    parts = label.split(',')
    score = 0
    for part in parts:
        attribute, value = part.split(':')
        value = float(value)
        weight = weights.get(attribute.strip(), 0)
        score += value * weight
    return score


def main(args):
    test_server_connection(args.host, args.port)
    inference_output = args.output_file

    exist = set()
    if os.path.exists(inference_output):
        with jsonlines.open(inference_output) as reader:
            for obj in tqdm(reader):
                exist.add(get_key(obj))

    fout = open(inference_output, "w", encoding="utf-8")
    print(f"Opened output file for appending: {inference_output}")

    # to warm up the jit
    _ = get_reward(["hello world!"], host=args.host, port=args.port, model_name=args.model_name)
    print("JIT warm-up completed. The reward is {}".format(_))
    #print('Routing Feature Shape is {}'.format(_[1].shape))
    # Stop here to debug the routing feature
#     sys.exit()

    all_samples, inputs = [], []
    sample_id_to_conversations = collections.defaultdict(list)
    subsets = []
    scores_chosen = []
    scores_rejected = []
    results = []

    attribute_weights = {
        "quality": 0.5,
        "toxicity": -0.5,
        "humor": 0,
        "creativity": 0,
        "helpfulness": 0.65,
        "correctness": 0.8,
        "coherence": 0.45,
        "complexity": 0.55,
        "verbosity": -0.4,
        "magpie_score": 0.5,
        "offset_bias": 0.5,
        "wild_guard": 0.5
    }

    with jsonlines.open(args.input_file) as reader:
        for obj in tqdm(reader):
            if get_key(obj) in exist:
                continue
            user = obj["mask"]
            turns = []
            text = SYSTEM_PROMPT_TEMPLATE.format(value=SYSTEM_PROMPT)
            for turn in obj["conversations"]:
                value = turn["value"]
                if turn["from"] == user:
                    text += USER_TURN_TEMPLATE.format(value=value)
                else:
                    text += ASSISTANT_TURN_TEMPLATE.format(value=value)
                if "label" in turn and turn["label"] is not None:
                    out_text = text + LABEL_PREFIX
                    turns.append(out_text)

            all_samples.append(turns)
            inputs.append(obj)
            sample_id_to_conversations[obj["sample_id"]].append(obj)
            subsets.append(obj["subset"])

    print(f"exist {len(exist)}, rest {len(inputs)}")
    if len(inputs) == 0:
        exit(0)

    for idx in trange(0, len(all_samples)):
        input = inputs[idx]
        sample = all_samples[idx]
        rewards_all = get_reward(
            sample, host=args.host, port=args.port, model_name=args.model_name
        )
        # print(rewards_all)

        t = 0
        for turn in input["conversations"]:
            if "label" in turn and turn["label"] is not None:
                reward = rewards_all[t]
                t += 1

                reward_string = ",".join(f"{a}:{r}" for a, r in zip(ALL_STEERLM_ATTRIBUTES, reward))
                turn["label"] = reward_string

        assert t == len(rewards_all)

        fout.write(json.dumps(input) + "\n")

    print("all annotations finished")
    fout.close()

    # Compare chosen and rejected responses and calculate accuracy per subset
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

        # Update subset results
        subset = chosen["subset"]
        subset_results[subset]['correct'] += int(result)
        subset_results[subset]['total'] += 1

    # Calculate overall accuracy
    accuracy = sum(results) / len(results) if results else 0
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Calculate and print accuracy per subset
    subset_results_str = ""
    for subset, counts in subset_results.items():
        subset_accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
        print(f"Subset {subset}: Accuracy = {subset_accuracy:.4f}")
        subset_results_str += f"Subset {subset}: Accuracy = {subset_accuracy:.4f}\n"

    # Log leaderboard aggregated results
    results_leaderboard_str = ""
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, subset_results)
        for section, score in results_leaderboard.items():
            results_leaderboard_str += f"Section {section}: Score = {score:.4f}\n"
        average_score = sum(results_leaderboard.values()) / len(results_leaderboard.keys())
        results_leaderboard_str += f"Average score across all sections: {average_score:.4f}\n"

    print(f"Overall Accuracy: {accuracy:.4f}")
    print(subset_results_str)
    print(results_leaderboard_str)

    # Save results to a file
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    res_filename = f"./data/rewardbench/results_{current_time}.txt"
    with open(res_filename, "w") as file:
        file.write(f"Overall Accuracy: {accuracy:.4f}\n")
        file.write(subset_results_str)
        file.write(results_leaderboard_str)

if __name__ == "__main__":
    main(prepare_args())
