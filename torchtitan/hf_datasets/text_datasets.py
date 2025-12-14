# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger

# Standard PyTorch ignore index for CrossEntropyLoss
IGNORE_INDEX = -100


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


def _load_chat_jsonl_dataset(dataset_path: str):
    """Load a JSONL dataset with chat messages format."""
    return load_dataset("json", data_files=dataset_path, split="train")


def _process_chat_jsonl_text(sample: dict[str, Any]) -> str:
    """
    Process chat JSONL dataset sample (legacy, for backward compatibility).
    Expected format: {"messages": [{"role": "system/user/assistant", "content": "..."}, ...]}
    Converts to a text format suitable for training.
    """
    messages = sample.get("messages", [])
    if not messages:
        return ""
    
    # Convert chat messages to a training-friendly format
    # Format: <|role|>content<|end|>...
    text_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text_parts.append(f"<|{role}|>\n{content}")
    
    return "\n".join(text_parts)


def process_chat_sft(
    sample: dict[str, Any],
    tokenizer: BaseTokenizer,
    assistant_role: str = "assistant",
) -> tuple[list[int], list[int]]:
    """
    Process chat JSONL for SFT training.
    Only computes loss on assistant responses (similar to fireworks SFT).
    
    Args:
        sample: Dict with "messages" list containing role/content pairs
        tokenizer: Tokenizer to encode text
        assistant_role: Role name for assistant messages (default: "assistant")
    
    Returns:
        (tokens, labels) where labels has IGNORE_INDEX for non-assistant tokens
    """
    messages = sample.get("messages", [])
    if not messages:
        return [], []
    
    all_tokens = []
    all_labels = []
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Format each message with role markers
        if i == 0:
            # First message gets BOS
            msg_text = f"<|{role}|>\n{content}"
            msg_tokens = tokenizer.encode(msg_text, add_bos=True, add_eos=False)
        elif i == len(messages) - 1:
            # Last message gets EOS
            msg_text = f"<|{role}|>\n{content}"
            msg_tokens = tokenizer.encode(msg_text, add_bos=False, add_eos=True)
        else:
            msg_text = f"<|{role}|>\n{content}"
            msg_tokens = tokenizer.encode(msg_text, add_bos=False, add_eos=False)
        
        all_tokens.extend(msg_tokens)
        
        # Only compute loss on assistant responses
        if role == assistant_role:
            # For assistant messages, use actual tokens as labels
            all_labels.extend(msg_tokens)
        else:
            # For system/user messages, mask out (ignore in loss)
            all_labels.extend([IGNORE_INDEX] * len(msg_tokens))
    
    return all_tokens, all_labels


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
    "chat_jsonl": DatasetConfig(
        path="",  # Will be provided via dataset_path
        loader=_load_chat_jsonl_dataset,
        sample_processor=_process_chat_jsonl_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []
        self._label_buffer: list[int] = []  # For SFT with masked labels

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Check if this is a chat format for SFT
                is_chat_sft = self.dataset_name == "chat_jsonl" and "messages" in sample
                
                if is_chat_sft:
                    # Use SFT processing with masked labels
                    sample_tokens, sample_labels = process_chat_sft(
                        sample, self._tokenizer, assistant_role="assistant"
                    )
                    self._token_buffer.extend(sample_tokens)
                    self._label_buffer.extend(sample_labels)
                else:
                    # Standard text processing (all tokens get loss)
                    sample_text = self._text_processor(sample)
                    sample_tokens = self._tokenizer.encode(
                        sample_text, add_bos=True, add_eos=True
                    )
                    self._token_buffer.extend(sample_tokens)
                    self._label_buffer.extend(sample_tokens)
                
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    tokens = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    labels = torch.LongTensor(self._label_buffer[:max_buffer_token_len])
                    
                    # update buffers to remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    self._label_buffer = self._label_buffer[max_buffer_token_len:]
                    
                    # input is tokens[:-1], label is labels[1:] (shifted by 1 for next-token prediction)
                    input_ids = tokens[:-1]
                    target_labels = labels[1:]
                    
                    yield {"input": input_ids}, target_labels

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]
        self._label_buffer = state_dict.get("label_buffer", [])

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {
            "token_buffer": self._token_buffer,
            "label_buffer": self._label_buffer,
        }

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets."""
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
