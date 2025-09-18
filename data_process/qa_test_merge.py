# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the QA dataset to parquet format
"""

import re
import os
import random
import datasets
from pathlib import Path

from verl.utils.hdfs_io import copy, makedirs
import argparse


from prompt_pool import PROMPT_TEMPLATE_QWEN, PROMPT_TEMPLATE_LLAMA


MATH_SOURCE_ALIASES = {"math", "gsm8k", "openai/gsm8k"}
MATH_CANONICAL_SOURCE = "openai/gsm8k"
MATH_FINAL_ANSWER_PATTERN = re.compile(r"####\s*(.*)")
MATH_ANSWER_STRIP_RE = re.compile(r"[\s\n]+")
MATH_INSTRUCTION = 'Please reason step by step and provide the final answer after "####".'


random.seed(42)


def make_prefix(dp, llm):
    question = dp['question']
    if llm == "qwen":
        prefix = PROMPT_TEMPLATE_QWEN.format_map({"question": question})
    elif llm == "llama":
        prefix = PROMPT_TEMPLATE_LLAMA.format_map({"question": question})
    else:
        raise NotImplementedError

    return prefix


def _is_math_source(data_source: str) -> bool:
    lowered = data_source.lower()
    return lowered in MATH_SOURCE_ALIASES or data_source == MATH_CANONICAL_SOURCE


def _canonicalize_source(data_source: str) -> str:
    return MATH_CANONICAL_SOURCE if _is_math_source(data_source) else data_source


def _format_question(raw_question: str, is_math: bool) -> str:
    question = raw_question.strip()
    if is_math:
        if '####' not in question:
            question = f"{question}\n\n{MATH_INSTRUCTION}"
        return question
    if question and question[-1] != '?':
        question += '?'
    return question


def _extract_question(example: dict, is_math: bool) -> str:
    if not is_math:
        return example['question']
    return example.get('question') or example.get('problem') or example.get('prompt') or ""


def _extract_math_ground_truth(answer_text: str) -> str:
    if not isinstance(answer_text, str):
        return ""
    match = MATH_FINAL_ANSWER_PATTERN.search(answer_text)
    if match:
        final_answer = match.group(1)
    else:
        final_answer = answer_text
    final_answer = final_answer.strip()
    final_answer = final_answer.split('\n')[0]
    final_answer = MATH_ANSWER_STRIP_RE.sub(' ', final_answer).strip()
    final_answer = final_answer.replace(',', '').replace('$', '')
    if final_answer.endswith('.'):
        final_answer = final_answer[:-1]
    return final_answer


# python data_process/qa_test_merge.py --data_sources nq,hotpotqa --model qwen
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_sources', default='nq,hotpotqa')
    parser.add_argument('--model', type=str, default='qwen')

    args = parser.parse_args()
    model_name = args.model.lower()
    data_sources = [src.strip() for src in args.data_sources.split(',') if src.strip()]
    all_dataset = []

    folder_path = Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    for data_source in data_sources:
        canonical_source = _canonicalize_source(data_source)
        is_math_source = canonical_source == MATH_CANONICAL_SOURCE

        if is_math_source:
            dataset = datasets.load_dataset('openai/gsm8k')
            if 'test' in dataset:
                print(f'Using the {canonical_source} test dataset...')
                test_dataset = dataset['test']
            else:
                print(f'Using the {canonical_source} validation dataset...')
                test_dataset = dataset['train']
        else:
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
            if 'test' in dataset:
                print(f'Using the {data_source} test dataset...')
                test_dataset = dataset['test']
            elif 'dev' in dataset:
                print(f'Using the {data_source} dev dataset...')
                test_dataset = dataset['dev']
            else:
                print(f'Using the {data_source} train dataset...')
                test_dataset = dataset['train']

        # random sample
        sample_size = min(len(test_dataset), 100)
        if sample_size < len(test_dataset):
            sampled_indices = random.sample(list(range(len(test_dataset))), sample_size)
            test_dataset = test_dataset.select(sampled_indices)

        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                question_text = _extract_question(example, is_math_source)
                question_text = _format_question(question_text, is_math_source)
                question = make_prefix({"question": question_text}, llm=model_name)

                if is_math_source:
                    solution = _extract_math_ground_truth(example.get('answer', ''))
                else:
                    solution = {
                        "target": example['golden_answers'],
                    }

                data = {
                    "data_source": canonical_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "math" if is_math_source else "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn

        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.to_parquet(os.path.join(local_dir, 'test_nh_{}.parquet'.format(model_name)))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
