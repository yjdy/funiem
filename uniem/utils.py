from __future__ import annotations

import os
import functools
import gc
import importlib
import json
import logging
from enum import Enum
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import Annotated, Any, Callable, Generator, Iterable, Optional, Sequence, Type, TypeVar, cast, Union, Sized

import torch
from torch.utils.data import DataLoader

import typer
import yaml
from accelerate.utils.memory import should_reduce_batch_size
from accelerate.utils import DummyOptim, DummyScheduler


from transformers import AutoModel, PreTrainedModel
from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset

from uniem.data import FinetuneDataset, FinetuneIterableDataset
from uniem.data import ScoredPairCollator, TripletCollator, PairCollator
from uniem.data_structures import RecordType

T = TypeVar('T')
logger = logging.getLogger(__name__)


class ConfigFileType(str, Enum):
    yaml = 'yaml'
    json = 'json'


def load_from_yaml(yaml_file: str | Path) -> dict[str, Any]:
    yaml_file = Path(yaml_file)
    if not yaml_file.exists():
        raise FileExistsError(f'File {yaml_file} does not exist')

    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def load_from_json(json_file: str | Path) -> dict[str, Any]:
    json_file = Path(json_file)
    if not json_file.exists():
        raise FileExistsError(f'File {json_file} does not exist')

    with open(json_file, 'r') as f:
        return json.load(f)


def load_config_file(config_file: str | Path, file_type: ConfigFileType | str | None = None) -> dict[str, Any]:
    config_file = Path(config_file)

    if file_type is None:
        file_name = config_file.name
        if file_name.endswith('.yaml') or file_name.endswith('.yml'):
            file_type = ConfigFileType.yaml
        elif file_name.endswith('.json'):
            file_type = ConfigFileType.json
        else:
            raise ValueError(f'Unknown config file format: {config_file}, only .yaml, .yml and .json are supported')
    else:
        file_type = ConfigFileType(file_type)

    if file_type == ConfigFileType.yaml:
            config = load_from_yaml(config_file)
    if file_type == ConfigFileType.json:
            config = load_from_json(config_file)
    return config


def _config_file_callback(ctx: typer.Context, param: typer.CallbackParam, param_value: Any):
    if param_value is None:
        return param_value
    try:
        config = load_config_file(param_value)
        ctx.default_map = ctx.default_map or {}
        ctx.default_map.update(config)
    except Exception as e:
        raise typer.BadParameter(str(e), ctx=ctx, param=param) from e
    return param_value


ConfigFile = Annotated[
    Optional[Path],
    typer.Option(..., callback=_config_file_callback, is_eager=True, help='Config file path, supports yaml and json'),
]


def load_hf_pretrained_model(
    model_name_or_path: str, model_class: str | None | Type[PreTrainedModel] | Type[AutoModel] = None
) -> PreTrainedModel:
    if model_class is None:
        model_class = AutoModel
    elif isinstance(model_class, str):
        transformers_module = importlib.import_module('transformers')
        model_class = getattr(transformers_module, model_class)

    model = model_class.from_pretrained(model_name_or_path)  # type: ignore
    model = cast(PreTrainedModel, model)
    return model


def create_adamw_optimizer(
    model: torch.nn.Module,
    lr: float = 0.001,
    weight_decay: float = 1e-2,
    betas=(0.9, 0.999),
    no_decay_keywords: Sequence[str] = ('bias', 'LayerNorm', 'layernorm'),
    dummy=False
):
    # default paramaters is same as admaw
    parameters = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay_keywords)],
            'weight_decay': weight_decay,
            'lr': lr
        },
        {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay_keywords)],
            'weight_decay': 0.0,
            'lr': lr
        },
    ]
    if not dummy:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr,betas=betas,weight_decay=weight_decay)
    else:
        # 使用deepspeed时需要使用dummy
        optimizer = DummyOptim(optimizer_grouped_parameters,lr=lr,weight_decay=weight_decay,betas=betas)
    return optimizer

def create_attention_mask_from_input_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return input_ids != pad_token_id


def generate_batch(data: Iterable[T], batch_size: int = 32) -> Generator[list[T], None, None]:
    iterator = iter(data)
    while batch := list(islice(iterator, batch_size)):
        yield batch

MapStyleDataset = Union[Sequence[dict], HFDataset]
IterableStyleDataset = Union[Iterable[dict], HFIterableDataset]
SupportedDataset = Union[MapStyleDataset, IterableStyleDataset]
SupportedDatasetDict = dict[str, SupportedDataset]


def create_finetune_datasets(raw_train_dataset, raw_validation_dataset,record_type,query_instruction='')\
        -> tuple[FinetuneDataset | FinetuneIterableDataset, FinetuneDataset | FinetuneIterableDataset | None]:
    if not isinstance(raw_train_dataset, Sized):
        raw_train_dataset = cast(IterableStyleDataset, raw_train_dataset)
        train_dataset = FinetuneIterableDataset(raw_train_dataset, record_type=record_type, query_instruction=query_instruction)
    else:
        train_dataset = FinetuneDataset(raw_train_dataset, record_type=record_type, query_instruction=query_instruction)

    if raw_validation_dataset is None:
        validation_dataset = None
    elif not isinstance(raw_validation_dataset, Sized):
        raw_validation_dataset = cast(IterableStyleDataset, raw_validation_dataset)
        validation_dataset = FinetuneIterableDataset(raw_validation_dataset, record_type=record_type, query_instruction=query_instruction)
    else:
        validation_dataset = FinetuneDataset(raw_validation_dataset, record_type=record_type, query_instruction=query_instruction)

    return train_dataset, validation_dataset


def create_dataloaders(
        train_dataset: FinetuneDataset | FinetuneIterableDataset,
        validation_dataset: FinetuneDataset | FinetuneIterableDataset | None,
        tokenizer,
        record_type,
        batch_size: int = 64,
        num_workers: int = 0,
        drop_last: bool = False,
        shuffle: bool = False,
        max_length: int | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    if record_type == RecordType.PAIR:
        data_collator = PairCollator(tokenizer=tokenizer, max_length=max_length)
    if record_type == RecordType.TRIPLET:
        data_collator = TripletCollator(tokenizer=tokenizer, max_length=max_length)
    if record_type == RecordType.SCORED_PAIR:
        data_collator = ScoredPairCollator(tokenizer=tokenizer, max_length=max_length)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    if validation_dataset is not None:
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        validation_dataloader = None
    return train_dataloader, validation_dataloader

def load_all_datasets_json(datasets_dir: Union[str, Path], need_validation=True, validation_name="validation.json") -> DatasetDict:
    data_files = []
    validation_file = ''
    for file_name in os.listdir(datasets_dir):
        if need_validation and validation_name==file_name:
            validation_file = os.path.join(datasets_dir, file_name)
        elif file_name.endswith("json"):
            data_files.append(os.path.join(datasets_dir, file_name))

    total_datasets = load_dataset("json", data_files=data_files)["train"]
    if need_validation and not validation_file:
        total_datasets = total_datasets.train_test_split(test_size=0.1,seed=42,shuffle=True)
        total_datasets["validation"] = total_datasets.pop("test")
    elif validation_file:
        total_datasets = DatasetDict(
            {"train":total_datasets,"validation":load_dataset("json", data_files=validation_file)["train"]}
        )
    return total_datasets


def split_dataset_dict(dataset_dict: dict[str, T]) -> tuple[T, T | None]:
    if isinstance(dataset_dict, dict):
        train_dataset = dataset_dict['train']
        if 'dev' in dataset_dict:
            validation_dataset = dataset_dict['dev']
        elif 'validation' in dataset_dict:
            validation_dataset = dataset_dict['validation']
        else:
            logger.warning(
                'No validation dataset found in dataset_dict, validation dataset key should be either "dev" or "validation"'
            )
            validation_dataset = None
    else:
        train_dataset = dataset_dict
        validation_dataset = None
    return train_dataset, validation_dataset


def find_executable_batch_size(function: Callable | None = None, starting_batch_size: int = 128):
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    @wraps(function)
    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        is_manually_passed_batch_size = 'batch_size' in kwargs

        if is_manually_passed_batch_size:
            return function(*args, **kwargs)
        else:
            while True:
                if batch_size == 0:
                    raise RuntimeError('No executable batch size found, reached zero.')
                try:
                    kwargs['batch_size'] = batch_size
                    return function(*args, **kwargs)
                except Exception as e:
                    if should_reduce_batch_size(e):
                        gc.collect()
                        torch.cuda.empty_cache()
                        batch_size //= 2
                        print('Reducing batch size to', batch_size)
                    else:
                        raise

    return decorator


def convert_number_to_readable_string(number: float) -> str:
    if number >= 1e9:
        return f'{number / 1e9:.1f}B'
    elif number >= 1e6:
        return f'{number / 1e6:.1f}M'
    elif number >= 1e3:
        return f'{number / 1e3:.1f}k'
    else:
        return str(number)
