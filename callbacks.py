import wandb
import time
from nemo.core import NeMoCallback
import glob
import os
import time
from abc import ABC
from typing import Callable, List, Union
from nemo.core.neural_types import NmTensor
from nemo.utils import get_checkpoint_from_dir, logging
from nemo.utils.app_state import AppState
from pathlib import Path

class WandbLogger(NeMoCallback):
    def __init__(
        self,
        step_freq: int = 100,
        save_freq: int = 1,
        tensors_to_log: List[Union[str, NmTensor]] = ["loss"],
        runid = None,
        args=None,
        log_epoch: bool = True,
        checkpoints_to_keep: int = 2,
        log_lr: bool = True,
        folder: Path = None,
        name: str = None,
        asr_model: str = None
    ):
        self._step_freq = step_freq
        self._tensors_to_log = tensors_to_log
        self._args = args
        self._last_epoch_start = None
        self._log_epoch = log_epoch
        self._log_lr = log_lr
        self._save_freq = save_freq
        self._folder = folder
        self._ckpt2keep = checkpoints_to_keep
        self._saved_ckpts = []
        self._name = name
        self._runid = runid
        self._asr_model = asr_model

    def on_action_start(self, state):
        if state["global_rank"] is None or state["global_rank"] == 0:
            if wandb.run is None:
                wandb.init(job_type='train',
                         id=self._runid,
                         tags=['train', 'nemo'],
                         group='train',
                         name=self._name,
                         project='asr',
                         entity='cprc')
                if self._args is not None:
                    logging.info('init wandb session and append args')
                    wandb.config.update(self._args)
            elif wandb.run is not None:
                logging.info("Re-using wandb session")
            else:
                logging.error("Could not import wandb. Did you install it (pip install --upgrade wandb)?")
                logging.info("Will not log data to weights and biases.")
                self._update_freq = -1

    def on_step_end(self, state):
        # log training metrics
        if state["global_rank"] is None or state["global_rank"] == 0:
            if state["step"] % self._step_freq == 0 and self._step_freq > 0:
                tensors_logged = {t: state["tensors"].get_tensor(t).cpu() for t in self._tensors_to_log}
                # Always log learning rate
                if self._log_lr:
                    tensors_logged['LR'] = state["optimizers"][0].param_groups[0]['lr']
                self._wandb_log(tensors_logged, state["step"])

    def on_epoch_start(self, state):
        if state["global_rank"] is None or state["global_rank"] == 0:
            self._last_epoch_start = time.time()

    def on_epoch_end(self, state):
        epoch = state["epoch"]
        if state["global_rank"] is None or state["global_rank"] == 0:
            if self._log_epoch:
                epoch_time = time.time() - self._last_epoch_start
                self._wandb_log({"epoch": epoch, "epoch_time": epoch_time}, state["step"])
        if self._save_freq > 0 and epoch % self._save_freq == 0 and epoch > 0:
            self.__save_to(self._folder, state, self._asr_model)

    @staticmethod
    def _wandb_log(tensors_logged, step):
        wandb.log(tensors_logged, step=step)

    def __save_to_wandb(self, model_file):
        upload_name = f'{self._name}_{wandb.run.id}'
        artifact = wandb.Artifact(
            type='acoustic_model',
            name=upload_name)
        upload_path = f'models/nemo/{upload_name}/{model_file.name}'

        # upload_blob(
        #     'cprc-dataset-bucket',
        #     str(model_file),
        #     upload_path
        # )
        # artifact.add_reference(f'gs://cprc-dataset-bucket/{upload_path}',
        #                     name=model_file.name)
        # wandb.run.log_artifact(artifact)

    def __save_to(self, path, state, asr_model):
        if state["global_rank"] is not None and state["global_rank"] != 0:
            return
        if not path.is_dir():
            logging.info(f"Creating {path} folder")
            os.makedirs(path, exist_ok=True)
        filename = f"am-{state['epoch']}.nemo"
        save_path = Path(path) / filename

        asr_model.save_to(str(save_path))
        self.__save_to_wandb(save_path)
    
        logging.info(f'Saved checkpoint: {save_path}')
        return save_path