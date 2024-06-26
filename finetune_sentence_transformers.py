# coding=utf-8
from __future__ import annotations

import os, sys

sys.path.append(os.getcwd())

from pathlib import Path
from typing import Annotated, Optional, Union, Callable

import typer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import DistributedDataParalleKwargs
from accelerate.utils import DummyScheduler # 为deepspeed准备


from transformers import AutoTokenizer, get_cosine_schedule_with_warmup


from uniem.model import (
    DataType,
    PoolingStrategy,
    create_uniem_embedder,
    create_uniem_embedder_trainer
)
from uniem.trainer import Trainer
from uniem.training_strategy import BitFitTrainging
from uniem.types import MixedPrecisionType
from uniem.utils import ConfigFile, convert_number_to_readable_string, create_adamw_optimizer
from uniem.utils import create_dataloaders, create_finetune_datasets, load_all_datasets_json
from uniem.data_structures import RecordType

import torch
import torch.nn as nn

def main(
    model_name_or_path: Union[str],
    datasets_dir: Union[str],
    # Model
    model_class: Annotated[Optional[str], typer.Option(rich_help_panel='Model')] = None,
    temperature: Annotated[float, typer.Option(rich_help_panel='Model')] = 0.05,
    pooling_strategy: Annotated[PoolingStrategy, typer.Option(rich_help_panel='Model')] = PoolingStrategy.last_mean,
    # Data
    record_type: Annotated[RecordType, typer.Option(rich_help_panel='Data')] = RecordType.SCORED_PAIR,
    batch_size: Annotated[int, typer.Option(rich_help_panel='Data')] = 32,
    with_instruction: Annotated[bool, typer.Option(rich_help_panel='Data')] = True,
    drop_last: Annotated[bool, typer.Option(rich_help_panel='Data')] = True,
    max_length: Annotated[int, typer.Option(rich_help_panel='Data')] = 512,
    query_instruction: str = '',
    # Optimizer
    lr: Annotated[float, typer.Option(rich_help_panel='Optimizer')] = 3e-5,
    weight_decay: Annotated[float, typer.Option(rich_help_panel='Optimizer')] = 1e-3,
    num_warmup_steps: Annotated[float, typer.Option(rich_help_panel='Optimizer')] = 0.05,
    # Trainer
    data_type: Annotated[DataType, typer.Option(rich_help_panel='Trainer')] = DataType.scored_pair,
    epochs: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 3,
    bitfit: Annotated[bool, typer.Option(rich_help_panel='Trainer')] = False,
    mixed_precision: Annotated[MixedPrecisionType, typer.Option(rich_help_panel='Trainer')] = MixedPrecisionType.no,
    gradient_accumulation_steps: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 1,
    save_on_epoch_end: Annotated[bool, typer.Option(rich_help_panel='Trainer')] = False,
    num_max_checkpoints: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 1,
    use_tensorboard: Annotated[bool, typer.Option(rich_help_panel='Trainer')] = False,
    num_workers: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 0,
    seed: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 42,
    output_dir: Annotated[Optional[Path], typer.Option(rich_help_panel='Trainer')] = None,
    logging_dir: Annotated[Optional[Path], typer.Option(rich_help_panel='Trainer')] = None,
    metric: Callable[[list, list], float] | None = None,
    # Config
    config_file: ConfigFile = None,
):
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
    if num_workers >= 1:
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    output_dir = Path(output_dir) if output_dir else Path('finetuned_model')
    if use_tensorboard:
        project_config = ProjectConfiguration(
            project_dir=str(output_dir),
            automatic_checkpoint_naming=True,
            total_limit=num_max_checkpoints,
            logging_dir=logging_dir if logging_dir else 'tensorboard'
        )
    else:
        project_config = ProjectConfiguration(
            project_dir=str(output_dir),
            automatic_checkpoint_naming=True,
            total_limit=num_max_checkpoints,
        )
    ddp_kwargs = DistributedDataParalleKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=mixed_precision.value,
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_config=project_config,
        log_with=['tensorboard'] if use_tensorboard else None,
        dispatch_batches=True,
        split_batches=True,
        kwargs_handlers = [ddp_kwargs]
    )
    accelerator.init_trackers(os.path.basename(model_name_or_path))
    accelerator.print(f'Parameters: {locals()}')

    set_seed(seed)
    accelerator.print(f'Start with seed: {seed}')
    accelerator.print(f'Output dir: {output_dir}')
    if config_file:
        accelerator.print(f'Config File: {config_file}')

    # DataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    with accelerator.main_process_first():
        all_datasets = load_all_datasets_json(datasets_dir)

    if "validation" in all_datasets:
        train_dataset, valid_dataset = create_finetune_datasets(all_datasets["train"], all_datasets["validation"],
                                                                record_type=record_type, query_instruction=query_instruction)
    else:
        train_dataset, valid_dataset = create_finetune_datasets(all_datasets["train"], None,
                                                                record_type=record_type,
                                                                query_instruction=query_instruction)


    train_dataloader, valid_dataloader = create_dataloaders(train_dataset,valid_dataset,tokenizer,
                                                            record_type=record_type,batch_size=batch_size,
                                                            shuffle=True, num_workers=num_workers,
                                                            drop_last=drop_last, max_length=max_length)

    # hack dataloader for distributed training
    train_dataloader.__dict__['batch_size'] = batch_size
    if valid_dataloader:
        valid_dataloader.__dict__['batch_size'] = batch_size

    embedder = create_uniem_embedder(
        model_name_or_path=model_name_or_path,
        model_class=model_class,
        pooling_strategy=pooling_strategy,
    )
    model = create_uniem_embedder_trainer(
        embedder=embedder,
        temperature=temperature,
        data_type=data_type,
    )
    if bitfit:
        # TO DO 训练策略配置化
        model = BitFitTrainging().apply_model(model)

    num_training_paramters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f'Number of training parameters: {convert_number_to_readable_string(num_training_paramters)}')
    embedder.encoder.config.pad_token_id = tokenizer.pad_token_id
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Optimizer & LRScheduler
    if (
        accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        optimizer = create_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = create_adamw_optimizer(model, lr=lr, weight_decay=weight_decay,dummy=True)
    total_steps = len(train_dataloader) * epochs
    if num_warmup_steps < 1:
        num_warmup_steps = int(num_warmup_steps * total_steps)

    if (
            accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(num_warmup_steps),
            num_training_steps=total_steps,
        )
    else:
        lr_scheduler = DummyScheduler(optimizer, num_warmup_steps=int(num_warmup_steps), total_num_steps=total_steps)
    valid_dataloader = accelerator.prepare(valid_dataloader) if valid_dataloader else None
    model, optimizer, lr_scheduler,train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader)

    def refresh_data(trainer: Trainer):
        train_dataset.create_or_refresh_data()

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=None,
        accelerator=accelerator,
        epochs=epochs,
        lr_scheduler=lr_scheduler,
        log_interval=10,
        save_on_epoch_end=save_on_epoch_end,
        metric=metric
    )
    accelerator.print(f'Start training for {epochs} epochs')
    trainer.train()

    # accelerator.wait_for_everyone()
    accelerator.print('Training finished')

    accelerator.print('Saving model')

    best_embedder = create_uniem_embedder(model_name_or_path=model_name_or_path,
                                          model_class=model_class,
                                          pooling_strategy=pooling_strategy)
    best_model = create_uniem_embedder_trainer(
        embedder=best_embedder,
        data_type=data_type,
        temperature=temperature
    )
    best_model.load_state_dict(torch.load(os.path.join(accelerator.project_configuration.project_dir, 'checkpoint.pt'))['model'])
    best_model.embedder.save_pretrained(os.path.join(output_dir, 'model'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'model'))


if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score

    main()
