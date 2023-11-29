from __future__ import annotations

import os
import gc
from typing import Any, Callable, Sequence, Sized

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        accelerator: Accelerator,
        validation_dataloader: DataLoader | None = None,
        epochs: int = 3,
        lr_scheduler: LRScheduler | None = None,
        log_interval: int = 50,
        save_on_epoch_end: bool = True,
        epoch_end_callbacks: Sequence[Callable[['Trainer'], None]] | None = None,
        metric = None,
        early_stopping_patience: int = 3
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_on_epoch_end = save_on_epoch_end

        self.metric = metric
        self.best_result = -1
        self.early_stopping_patience = early_stopping_patience
        self.current_patience = early_stopping_patience

        self.train_loss_tracker = LossTracker()
        self.validation_loss_tracker = LossTracker()
        if isinstance(self.train_dataloader.dataset, Sized):
            num_steps_per_epoch = len(self.train_dataloader)
        else:
            num_steps_per_epoch = None
        self.progress_bar = DistributedTqdmProgressBar(self.epochs, num_steps_per_epoch=num_steps_per_epoch)
        self.epoch_end_callbacks = epoch_end_callbacks or []
        self.current_step = 0

    def train(self, test_first=True):
        if test_first and self.validation_dataloader:
            validation_loss, validation_result = evaluate(
                self.model,
                self.validation_dataloader,
                accelerator=self.accelerator,
                loss_tracker=self.train_loss_tracker,
                metric=self.metric
            )
            if validation_result:
                self.accelerator.print(f"Validation loss: {validation_loss}, metric_result: {validation_result: .6f}")
                self.best_result = validation_result
            
        for current_epoch in range(1, self.epochs + 1):
            self.model.train()
            self.progress_bar.on_epoch_start()

            for batch_index, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    batch_output = self.model(**batch)
                    loss = batch_output['loss']
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if self.lr_scheduler is not None and not self.accelerator.optimizer_step_was_skipped:
                        self.lr_scheduler.step()
                    self.train_loss_tracker.update(loss)

                self.progress_bar.update()
                self.current_step += 1
                if batch_index % self.log_interval == 0:
                    self.log_metrics(
                        {'loss': self.train_loss_tracker.loss},
                        step=self.current_step,
                    )
            gc.collect()
            torch.cuda.empty_cache()

            train_metrics = self.add_prefix({'loss': self.train_loss_tracker.loss}, 'train')
            self.accelerator.log(train_metrics, step=current_epoch)
            self.train_loss_tracker.on_epoch_end()
            self.progress_bar.on_epoch_end()

            if self.validation_dataloader:
                validation_loss, validation_result = evaluate(
                    self.model,
                    self.validation_dataloader,
                    accelerator=self.accelerator,
                    loss_tracker=self.train_loss_tracker,
                    metric=self.metric
                )
                
                if validation_result:
                    validation_metrics = self.add_prefix({'loss': validation_loss,'metric':validation_result}, 'validation')
                    self.accelerator.print(f"Validation loss: {validation_loss}, metric_result: {validation_result: .6f}")
                    if self.best_result < validation_result:
                        self.best_result = validation_result
                        self.current_patience = self.early_stopping_patience
                        if self.accelerator.is_main_process:
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            self.accelerator.save(
                                "model":unwrapped_model.state_dict(),
                                "optimizer": self.optimizer.state_dict(),
                                os.path.join(self.accelerator.project_configuration.project_dir, "checpoint.pt")
                            )
                    else:
                        self.current_patience -= 1
                        if self.current_patience < 0:
                            self.accelerator.print("Early Stopping")
                            self.accelerator.set_trigger()
                    if self.accelerator.check_trigger():
                break
            else:
                validation_metrics = self.add_prefix({'loss': validation_loss}, 'validation')
                self.accelerator.print(f'Epoch {current_epoch} Validation loss: {validation_loss:.4f}')
                self.accelerator.log(validation_metrics, step=current_epoch)
            

            # if self.save_on_epoch_end:
            #     self.accelerator.save_state(self.get_checkpoint_dir())

            # if self.epoch_end_callbacks:
            #     for callback in self.epoch_end_callbacks:
            #         callback(self)

        self.accelerator.end_training()

    def log_metrics(self, metrics: dict[str, float], step: int):
        self.accelerator.log(metrics, step=step)
        self.progress_bar.show_metrics(metrics)

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f'{prefix}/{k}': v for k, v in values.items()}

    def get_checkpoint_dir(self):
        # COPY FROM accelerator to fix Checkpoint bug
        self.accelerator.project_configuration.automatic_checkpoint_naming = False
        output_dir = os.path.join(self.accelerator.project_dir, 'checkpoints')
        if self.accelerator.is_local_main_process:
            os.makedirs(output_dir, exist_ok=True)
            folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
            if self.accelerator.project_configuration.total_limit is not None and (
                len(folders) + 1 > self.accelerator.project_configuration.total_limit
            ):

                def _inner(folder):
                    return list(map(int, re.findall(r'[\/]?([0-9]+)(?=[^\/]*$)', folder)))[0]

                folders.sort(key=_inner)
                logger.warning(
                    f'Deleting {len(folders) + 1 - self.accelerator.project_configuration.total_limit}'
                    'checkpoints to make room for new checkpoint.'
                )
                for folder in folders[: len(folders) + 1 - self.accelerator.project_configuration.total_limit]:
                    shutil.rmtree(folder)

        output_dir = os.path.join(output_dir, f'checkpoint_{self.accelerator.save_iteration}')
        if self.accelerator.is_local_main_process:
            os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving current state to {output_dir}')
        return output_dir


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    accelerator = None,
    loss_tracker: LossTracker | None = None,
    metric = None
):
    model = model.eval()
    if metric:
        predicts = []
        labels = []
    loss_tracker = loss_tracker or LossTracker()
    for batch in dataloader:
        with torch.inference_mode():
            batch_output = accelerator.unwrap_model(model)(**batch)
            loss_tracker.update(batch_output['loss'])
            if metric:
                predicts.extend(batch_output["predict_labels"].detach().cpu().numpy())
                labels.extend(batch_output["lables"].detach().cpu().numpy())
    if metric:
        metric_result = metric(labels, predicts)
    else:
        metric_result = None
    loss = loss_tracker.loss
    loss_tracker.on_epoch_end()
    return loss, metric_result


class DummyProgressBar:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_description(self, description: str) -> None:
        pass


class DistributedTqdmProgressBar:
    def __init__(self, epochs: int, num_steps_per_epoch: int | None, **kwargs) -> None:
        self.accelerator = Accelerator()
        self.epochs = epochs
        self.current_epoch = 1
        self.num_steps_per_epoch = num_steps_per_epoch
        self.tqdm_kwargs = kwargs

    def on_epoch_start(self):
        if self.accelerator.is_main_process:
            self.progress_bar = tqdm(total=self.num_steps_per_epoch, **self.tqdm_kwargs)
        else:
            self.progress_bar = DummyProgressBar()

    def update(self, n: int = 1) -> None:
        self.progress_bar.update(n)

    def close(self) -> None:
        self.progress_bar.close()

    def on_epoch_end(self) -> None:
        self.current_epoch += 1
        self.progress_bar.close()

    def show_metrics(self, metrics: dict[str, float]) -> None:
        description = f'Epoch {self.current_epoch}/{self.epochs}'
        for name, score in metrics.items():
            description += f' - {name}: {score:.4f}'
        self.progress_bar.set_description(description)


class LossTracker:
    def __init__(
        self,
        ndigits=4,
    ) -> None:
        self.ndigits = ndigits
        self._loss: float = 0.0
        self.loss_count: int = 0
        self.history: list[float] = []

    def update(self, loss_tensor: torch.Tensor):
        loss = loss_tensor.item()
        self._loss = (self._loss * self.loss_count + loss) / (self.loss_count + 1)
        self.loss_count += 1

    def reset(self):
        self._loss = 0
        self.loss_count = 0

    def on_epoch_end(self, reset: bool = True):
        self.history.append(self.loss)
        if reset:
            self.reset()

    @property
    def loss(self) -> float:
        return round(float(self._loss), self.ndigits)
