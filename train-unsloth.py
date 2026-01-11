import os
import re
import json
import argparse
import numpy as np
from functools import partial, update_wrapper

from unsloth import FastLanguageModel
from datasets import Dataset, DatasetDict
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"


def init_unsloth_model(model_name, max_model_len, lora_rank, gpu_util):

    model, tokenizer = FastLanguageModel.from_pretrained(model_name = model_name,
                                                         max_seq_length = max_model_len,
                                                         load_in_4bit = False, # False for LoRA 16bit
                                                         fast_inference = True, # Enable vLLM fast inference
                                                         max_lora_rank = lora_rank,
                                                         gpu_memory_utilization = gpu_util)

    model = FastLanguageModel.get_peft_model(model,
                                             r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                                             target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], # Remove QKVO if out of memory
                                             lora_alpha = lora_rank*2,
                                             use_gradient_checkpointing = 'unsloth', # Enable long context finetuning
                                             random_state = 3407)
    
    return model, tokenizer


def load_data(dataset_path, tokenizer, max_model_len, max_tokens):

    with open(dataset_path) as fr:
        data = json.load(fr)

    split_names = list(data.keys())

    total_instances = sum(len(data[split]) for split in split_names)
    dropped_instances = 0

    revised_data = {split: [] for split in split_names}

    for split in ['train', 'test']:
        for instance in data[split]:
            input_len = len(tokenizer.apply_chat_template(instance['prompt'], add_generation_prompt=True, tokenize=True))
            if input_len <= (max_model_len - max_tokens - 1):
                revised_data[split].append(instance)
            else:
                dropped_instances += 1

    dataset = DatasetDict({"train": Dataset.from_list(revised_data["train"]),
                           "test": Dataset.from_list(revised_data["test"])})

    print(f"{dropped_instances}/{total_instances} instances dropped due to exceeding context length.")

    return dataset


def compute_score(completions, labels, score_sets, max_tokens, tokenizer, **kwargs):
    """
    Reward function that scores LLM outputs which contain a score in <score></score> format.
    The reward is based on:
      - Correct use of the format
      - Presence of a valid numeric value
      - Equality of value and expected score
    """
    def length_penalty(length):
        lower = 0.25 * max_tokens
        upper = 0.75 * max_tokens
        if length < lower:
            penalty = (lower - length) ** 2 / lower ** 2
        elif length > upper:
            penalty = (length - upper) ** 2  / (max_tokens - upper) ** 2
        else:
            penalty = 0

        return np.clip(penalty, 0, 1)

    # Pattern to find <score>...</score> with any number (integer or float)
    def get_reward(s, label, score_set, length):
        try:
            if float(s.strip()) in score_set:
                return float(float(s.strip()) == label) + 0.5 - length_penalty(length)
            else:
                return 0.25
        except (ValueError, TypeError):
            return 0

    pattern = re.compile(r"<score>([^<]+)</score>")

    matches = [re.search(pattern, completion[0]['content']) for completion in completions]

    lengths = [len(tokenizer.encode(completion[0]['content'])) for completion in completions]

    rewards = np.array([get_reward(m.group(1), label, score_set, length) if m else -0.5 for m, label, score_set, length in zip(matches, labels, score_sets, lengths)])

    print(f"\nEXAMPLE FROM BATCH\n\nCompletion: {completions[0][0]['content']}\n\nLabel: {labels[0]}\n\nReward: {rewards[0]}\n\n")

    return rewards


def init_trainer(exp_name, model, tokenizer, dataset, temp, top_p, max_model_len,
                 batch_size, max_tokens, rollout, output_path, logging_steps=100, save_steps=500, epochs=1):

    training_args = GRPOConfig(run_name=exp_name,
                               temperature=temp,
                               top_p=top_p,
                               beta=0.001,
                               learning_rate=5e-6,
                               weight_decay=0.1,
                               warmup_ratio=0.1,
                               lr_scheduler_type="cosine",
                               optim="adamw_8bit",
                               logging_steps=logging_steps,
                               per_device_train_batch_size=batch_size,
                               steps_per_generation=4,
                               gradient_accumulation_steps=4,
                               num_generations=rollout,
                               max_prompt_length=max_model_len-max_tokens,
                               max_completion_length=max_tokens,
                               num_train_epochs = epochs,
                               save_steps=save_steps,
                               save_total_limit=1,
                               output_dir=os.path.join(output_path,exp_name))

    reward_fn = partial(compute_score, max_tokens=max_tokens, tokenizer=tokenizer)
    update_wrapper(reward_fn, compute_score)

    trainer = GRPOTrainer(model=model,
                          processing_class=tokenizer,
                          reward_funcs=reward_fn,
                          args=training_args,
                          train_dataset=dataset['train'])

    return trainer


def main(args):

    model, tokenizer = init_unsloth_model(args.model_name, args.max_model_len, args.lora_rank, args.vllm_gpu_memory_utilization)

    dataset = load_data(args.dataset_file, tokenizer, args.max_model_len, args.max_tokens)

    trainer = init_trainer(args.exp_name, model, tokenizer, dataset, args.temp, args.top_p, args.max_model_len,
                           args.batch_size, args.max_tokens, args.rollout, args.output_path)

    trainer.train()

    model.save_pretrained_merged(os.path.join(args.output_path, args.exp_name, 'final_model'), tokenizer, save_method="merged_16bit")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--dataset_file', required=True, type=str)
    parser.add_argument('--vllm_gpu_memory_utilization', default=0.6, type=float)
    parser.add_argument('--max_model_len', default=4096, type=int)
    parser.add_argument('--lora_rank', default=64, type=int)
    parser.add_argument('--max_tokens', default=512, type=int)
    parser.add_argument('--temp', default=1, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--rollout', default=4, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--output_path', default='train-runs', type=str)

    arguments = parser.parse_args()
    main(arguments)
