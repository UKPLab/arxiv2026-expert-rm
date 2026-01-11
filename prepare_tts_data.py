import re
import json
import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


def create_tts_prompts(inference_data, tokenizer):

    new_prompts = []
    dropped_index = []

    for row in inference_data.itertuples():
        if row.reward_1 not in [0.5, 1.5]:
            dropped_index.append(row.Index)
            continue
        else:

            if "<think>" in row.output_1 and "</think>" in row.output_1:
                match_reasoning = re.search(r"<think>(.*?)</think>", row.output_1, flags=re.DOTALL)
                reason_end_token = "</think>"
            else:
                match_reasoning = re.search(r"<reasoning>(.*?)</reasoning>", row.output_1, flags=re.DOTALL)
                reason_end_token = "</reasoning>"

            match_score = re.search(r"<score>(.*?)</score>", row.output_1, flags=re.DOTALL)
            if (not match_reasoning) or (not match_score):
                dropped_index.append(row.Index)
                continue

            # Scores should be at the outside of the reasoning part
            if f"<score>{match_score.group(1).strip()}</score>" in match_reasoning.group(1).strip():
                dropped_index.append(row.Index)
                continue

            reasoning = row.output_1[:row.output_1.index(reason_end_token)]

            new_reasoning = (f"{reasoning}\n\nWait. Let me check query and criteria to assess the reasoning before my final answer. ")

            messages = list(row.prompt) + [{'role': 'assistant', 'content': new_reasoning}]

            new_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False,
                                                             add_generation_prompt=False, continue_final_message=True))

    tts_data = inference_data.drop(dropped_index)
    tts_data['prompt'] = tts_data['prompt'].map(np.ndarray.tolist)
    tts_data['score_sets'] = tts_data['score_sets'].map(np.ndarray.tolist)
    tts_data = tts_data.rename(columns={"prompt": "init_prompt"})
    tts_data['prompt'] = new_prompts

    return tts_data


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    rev_util_outputs = pd.read_parquet(args.rev_util_outputs)
    rw_gen_outputs = pd.read_parquet(args.rw_gen_outputs)

    tts_rev_util_data = create_tts_prompts(rev_util_outputs, tokenizer)
    tts_rw_gen_output = create_tts_prompts(rw_gen_outputs, tokenizer)

    tts_data = pd.concat([tts_rev_util_data, tts_rw_gen_output], axis=0)
    tts_data = tts_data.sample(frac=1, random_state=42).reset_index(drop=True)

    output = {'train': tts_data.to_dict(orient='records'), 'test': []}

    with open(args.output_path, 'w') as fw:
        json.dump(output, fw, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--rev_util_outputs', required=True, type=str)
    parser.add_argument('--rw_gen_outputs', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)

    arguments = parser.parse_args()
    main(arguments)