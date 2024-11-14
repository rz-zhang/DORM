import argparse
import collections
import json
import os
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



def test_server_connection(host, port, message="hello world!"):
    try:
        with socket.create_connection((host, port)) as sock:
            sock.sendall(message.encode('utf-8'))
            response = sock.recv(1024)
            print(f"Received: {response.decode('utf-8')}")
            print(f"Received: {response.decode('utf-8')}")
    except socket.error as e:
        print(f"Unable to connect to {host}:{port}. Error: {e}")



def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Helper function for immediately logging RewardBench scores.
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
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
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--model_name", type=str, default="reward_model")
#     parser.add_argument("--add-eos", action="store_true")
    # parser.add_argument("--model", type=str, required=True, help="Model name for logging")
    # parser.add_argument("--model_type", type=str, required=True, help="Model type for logging")
    # parser.add_argument("--chat_template", type=str, required=True, help="Chat template for logging")
    parser.add_argument("--pref_sets", action="store_true", help="Flag to indicate preference sets evaluation")
    return parser.parse_args()


def get_reward(
    sentences: List[str], host="localhost", port=5555, model_name="reward_model",
):
    sentences = _str_list2numpy(sentences)

    futures = []

    with FuturesModelClient(f"{host}:{port}", model_name) as client:
        for sen in np.split(sentences, sentences.shape[0]):
#             add_EOS_arr = np.ones_like(sen, dtype=bool) * add_EOS
            future = client.infer_batch(sentences=sen)
            futures.append(future)

    all_result_dicts = [f.result() for f in futures]

    all_rewards, all_exceeded = [], []

    for output_dict in all_result_dicts:
        print(output_dict)
        reward_out = output_dict["rewards"].flatten().tolist()

        all_rewards.append(reward_out)
#         all_exceeded += output_dict["exceeded"].tolist()

    return all_rewards


def get_key(l):
    convs = [c["value"] for c in l["conversations"]]
    return "".join(convs)


def main(args):
    test_server_connection(args.host, args.port)
    inference_output = args.output_file

    exist = set()
    if os.path.exists(inference_output):
        with jsonlines.open(inference_output) as reader:
            for obj in tqdm(reader):
                exist.add(get_key(obj))

    fout = open(inference_output, "a", encoding="utf-8")
    print(f"Opened output file for appending: {inference_output}")

    # to warm up the jit
    _ = get_reward(["hello world!"], host=args.host, port=args.port, model_name=args.model_name)
    print("JIT warm-up completed. The reward is {}".format(_))

    all_samples, inputs = [], []
    sample_id_to_conversations = collections.defaultdict(list)
    subsets = []
    scores_chosen = []
    scores_rejected = []
    results = []

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

    for idx in trange(0, len(all_samples[:10])):
        input = inputs[idx]
        sample = all_samples[idx]
        rewards_all = get_reward(
            sample, host=args.host, port=args.port, model_name=args.model_name
        )
        print(rewards_all)

        t = 0
        for turn in input["conversations"]:
            if "label" in turn and turn["label"] is not None:
                reward = rewards_all[t]
                t += 1

                reward_string = ",".join(f"{a}:{r}" for a, r in zip(ALL_STEERLM_ATTRIBUTES, reward))
                turn["label"] = reward_string

        assert t == len(rewards_all)
        print(turn["label"])

        fout.write(json.dumps(input) + "\n")

    print("all annotations finished")
    fout.close()

    # Compare chosen and rejected responses
    for sample_id, conversations in sample_id_to_conversations.items():
        chosen_label = None
        rejected_label = None
        subset = None
        for conversation in conversations:
            if conversation["conversations"][1]["value"] == conversation["chosen"]:
                chosen_label = conversation["conversations"][1]["label"]
                subset = conversation["subset"]
                chosen_score = sum(map(float, chosen_label.split(",")[0].split(":")[1:])) / len(ALL_STEERLM_ATTRIBUTES)
                scores_chosen.append(chosen_score)
                if rejected_label:
                    rejected_score = sum(map(float, rejected_label.split(",")[0].split(":")[1:])) / len(ALL_STEERLM_ATTRIBUTES)
                    results.append(chosen_score > rejected_score)
            elif conversation["conversations"][1]["value"] == conversation["rejected"]:
                rejected_label = conversation["conversations"][1]["label"]
                rejected_score = sum(map(float, rejected_label.split(",")[0].split(":")[1:])) / len(ALL_STEERLM_ATTRIBUTES)
                scores_rejected.append(rejected_score)

        if chosen_label and rejected_label:
            print(f"Sample ID: {sample_id}, Subset: {subset}")
            print(f"Chosen Response Label: {chosen_label}")
            print(f"Rejected Response Label: {rejected_label}")
            print()

    # Add columns for results, subsets, scores_chosen, and scores_rejected
    out_dataset = collections.defaultdict(list)
    out_dataset["results"] = results
    out_dataset["subset"] = subsets
    out_dataset["scores_chosen"] = scores_chosen
    out_dataset["scores_rejected"] = scores_rejected

    # Print per subset and log into results_grouped
    present_subsets = set(subsets)
    results_grouped = {}

    for subset in present_subsets:
        subset_results = [r for r, s in zip(out_dataset["results"], out_dataset["subset"]) if s == subset]
        num_correct = sum(subset_results)
        num_total = len(subset_results)
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total


    # Log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)


if __name__ == "__main__":
    main(prepare_args())