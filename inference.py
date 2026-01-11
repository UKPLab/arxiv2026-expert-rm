import re
import os
import argparse
import torch
from vllm import LLM
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict


def load_data(dataset_path):

    with open(dataset_path) as fr:
        data = json.load(fr)

    dataset = DatasetDict({"train": Dataset.from_list(data["train"]),
                           "test": Dataset.from_list(data["test"])})
    return dataset


def compute_score(completions, labels, score_sets):
    """
    Reward function that scores LLM outputs which contain a score in <score></score> format.
    The reward is based on:
      - Correct use of the format
      - Presence of a valid numeric value
      - Equality of value and expected score
    """
    # Pattern to find <score>...</score> with any number (integer or float)

    def get_reward(s, label, score_set):
        try:
            if float(s.strip()) in score_set:
                return float(float(s.strip()) == label) + 0.5
            else:
                return 0.25
        except (ValueError, TypeError):
            return 0

    pattern = re.compile(r"<score>([^<]+)</score>")

    matches = [re.search(pattern, completion.outputs[0].text) for completion in completions]

    rewards = np.array([get_reward(m.group(1), label, score_set) if m else -0.5 for m, label, score_set in zip(matches, labels, score_sets)])

    print(f"\nEXAMPLE FROM BATCH\n\nCompletion: {completions[0].outputs[0].text}\n\nLabel:{labels[0]}\n\nReward: {rewards[0]}\n\n")

    return rewards


def init_model(model_name, max_model_len, max_tokens, temp, top_p, gpu_util=0.9):

    model = LLM(model=model_name, dtype=torch.bfloat16, max_model_len=max_model_len, trust_remote_code=True, gpu_memory_utilization=gpu_util)

    sampling_params = model.get_default_sampling_params()
    sampling_params.max_tokens = max_tokens
    sampling_params.temperature = temp
    sampling_params.top_p = top_p

    return model, sampling_params


def inference(batch, model, sampling_params):
    completions = model.chat(batch['prompt'], sampling_params)
    return {'output': [completion.outputs[0].text for completion in completions],
            'reward': compute_score(completions, batch['labels'], batch['score_sets'])}


def main(args):

    eval_dataset = load_data(args.dataset_file)['test']
    model, sampling_params = init_model(args.model_name, args.max_model_len, args.max_tokens, args.temp, args.top_p)

    if not os.path.exists(os.path.join(args.output_path, args.exp_name)):
        os.makedirs(os.path.join(args.output_path, args.exp_name))

    for task in set(eval_dataset['task']):

        task_dataset = eval_dataset.filter(lambda ex: ex['task'] == task)

        results = {aspect: {'rollout_reward_dist': [], 'rollout_sums': [], 'rollout_means': [], 'rollout_stds': []}
                      for aspect in set(task_dataset['aspect'])}

        results['whole_task'] = {'rollout_reward_dist': [], 'rollout_sums': [], 'rollout_means': [], 'rollout_stds': []}

        outputs = []

        for turn in range(args.rollout):

            ds = task_dataset.map(inference, fn_kwargs={'model':model, 'sampling_params':sampling_params}, batched=True,
                                  batch_size=args.batch_size).to_pandas()

            outputs.append(ds[['output', 'reward']].rename(columns={'output': f'output_{turn+1}', 'reward': f'reward_{turn+1}'}))

            results['whole_task']['rollout_reward_dist'].append({reward: sum(ds['reward'] == reward) for reward in [-0.5, 0.0, 0.25, 0.5, 1.5]})
            results['whole_task']['rollout_sums'].append(ds['reward'].sum())
            results['whole_task']['rollout_means'].append(ds['reward'].mean())
            results['whole_task']['rollout_stds'].append(ds['reward'].std())

            for aspect in [k for k in results.keys() if k != 'whole_task']:
                aspect_subset = ds[ds['aspect'] == aspect]
                results[aspect]['rollout_reward_dist'].append({reward: sum(aspect_subset['reward']==reward) for reward in [-0.5, 0.0, 0.25, 0.5, 1.5]})
                results[aspect]['rollout_sums'].append(aspect_subset['reward'].sum())
                results[aspect]['rollout_means'].append(aspect_subset['reward'].mean())
                results[aspect]['rollout_stds'].append(aspect_subset['reward'].std())

        results['whole_task']['overall_task_reward_dist'] = {k: sum(d[k] for d in results['whole_task']['rollout_reward_dist']) for k in results['whole_task']['rollout_reward_dist'][0]}
        results['whole_task']['overall_task_reward_mean'] = np.mean(results['whole_task']['rollout_means'])
        results['whole_task']['task_reward_mean_stds'] = np.mean(results['whole_task']['rollout_stds'])

        for aspect in [k for k in results.keys() if k != 'whole_task']:
            results[aspect]['overall_aspect_reward_dist'] = {k: sum(d[k] for d in results[aspect]['rollout_reward_dist']) for k in results[aspect]['rollout_reward_dist'][0]}
            results[aspect]['overall_aspect_reward_mean'] = np.mean(results[aspect]['rollout_means'])
            results[aspect]['aspect_reward_mean_stds'] = np.mean(results[aspect]['rollout_stds'])

        final_outputs = pd.concat([ds.drop(columns=['output', 'reward']).reset_index(drop=True)] + outputs, axis=1)

        task_dict = vars(args)
        task_dict['task'] = task
        task_dict['results'] = results

        with open(os.path.join(args.output_path, args.exp_name, f"{task}_results.json"), 'w') as fw:
            json.dump(task_dict, fw, indent=4)

        final_outputs.to_parquet(os.path.join(args.output_path, args.exp_name, f"{task}_outputs.parquet"), index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--dataset_file', required=True, type=str)
    parser.add_argument('--max_model_len', default=32768, type=int)
    parser.add_argument('--max_tokens', default=2048, type=int)
    parser.add_argument('--temp', default=1, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--rollout', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--output_path', required=True, type=str)

    arguments = parser.parse_args()
    main(arguments)
