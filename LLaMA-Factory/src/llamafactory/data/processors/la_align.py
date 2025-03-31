# Copyright 2024 the LlamaFactory team.
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import infer_seqlen

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = get_logger(__name__)


def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    messages = template.mm_plugin.process_messages(prompt + response, images, videos, processor)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)
    encoded_pairs = template.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if template.efficient_eos else 0)
    if mask_history:
        encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:
            break

        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if train_on_prompt:
            source_label = source_ids
        elif template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != 0:  # train on the last turn only
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if mask_history:  # reversed sequences
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return input_ids, labels


def preprocess_la_align_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    encoder_tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i]))
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        # Change to Langbridge format
        # 找到最后一个 -100 的索引
        last_index = len(labels) - 1 - labels[::-1].index(-100)
        # Instruction input_ids 获取最后一个 -100 之前的所有元素
        ins_input_ids = input_ids[:last_index + 1]
        # input_ids 获取最后一个 -100 之后的所有元素
        input_ids = input_ids[last_index + 1:]
        labels = labels[last_index + 1:]
        # convert list to tensor
        ins_input_ids = torch.tensor(ins_input_ids, dtype=torch.int64)
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        # add bos token id to labels
        bos_token_id_tensor = torch.tensor([tokenizer.bos_token_id])
        ins_input_ids = torch.cat((bos_token_id_tensor, ins_input_ids), dim=0)
        # labels = torch.cat((bos_token_id_tensor, labels), dim=0)
        # original code
        model_inputs["input_ids"].append(input_ids)
        model_inputs["ins_input_ids"].append(ins_input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["ins_attention_mask"].append([1] * len(ins_input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])
        # process encoder input
        encoder_tokenizer.add_bos_token = False
        encoder_tokenizer.add_eos_token = False
        encoder_inputs = encoder_tokenizer(examples["_enc_input"][i], padding=True, truncation=True, max_length=data_args.max_length_enc, return_tensors='pt')
        model_inputs["encoder_input_ids"].append(encoder_inputs["input_ids"][0])
        model_inputs["encoder_attention_mask"].append(encoder_inputs["attention_mask"][0])
        # process parallel data for align training
        if examples["_sent_src"][i] is not None and examples["_sent_tgt"][i] is not None:
            # process encoder input
            encoder_src_inputs = encoder_tokenizer(examples["_sent_src"][i], padding=True, truncation=True, max_length=data_args.max_length_enc, return_tensors='pt')
            model_inputs["encoder_src_ids"].append(encoder_src_inputs["input_ids"][0])
            model_inputs["encoder_src_attention_mask"].append(encoder_src_inputs["attention_mask"][0])
            model_inputs["src_lang"].append(examples["_src_lang"][i])
            encoder_tgt_inputs = encoder_tokenizer(examples["_sent_tgt"][i], padding=True, truncation=True, max_length=data_args.max_length_enc, return_tensors='pt')
            model_inputs["encoder_tgt_ids"].append(encoder_tgt_inputs["input_ids"][0])
            model_inputs["encoder_tgt_attention_mask"].append(encoder_tgt_inputs["attention_mask"][0])
            model_inputs["tgt_lang"].append(examples["_tgt_lang"][i])

    return model_inputs


def print_la_align_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer", encoder_tokenizer: "PreTrainedTokenizer") -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("ins_input_ids:\n{}".format(example["ins_input_ids"]))
    print("ins_inputs:\n{}".format(tokenizer.decode(example["ins_input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))
    print("encoder_input_ids:\n{}".format(example["encoder_input_ids"]))
    print("encoder_inputs:\n{}".format(encoder_tokenizer.decode(example["encoder_input_ids"], skip_special_tokens=False)))
    if "encoder_src_ids" in example.keys():
        print("encoder_src_ids:\n{}".format(example["encoder_src_ids"]))
        print("encoder_src:\n{}".format(encoder_tokenizer.decode(example["encoder_src_ids"], skip_special_tokens=False)))
        print("source language:\n{}".format(example["src_lang"]))
        print("encoder_tgt_ids:\n{}".format(example["encoder_tgt_ids"]))
        print("encoder_tgt:\n{}".format(encoder_tokenizer.decode(example["encoder_tgt_ids"], skip_special_tokens=False)))
        print("target language:\n{}".format(example["tgt_lang"]))
