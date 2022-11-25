from pathlib import Path

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


class Checkpoint(ModelCheckpoint):
    def __init__(self) -> None:
        """A callback to checkpoint files during training."""
        filename = "{epoch:03d}-{val_loss:.5f}"
        super().__init__(save_top_k=-1, filename=filename)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage == "fit":
            if trainer.fast_dev_run:
                return

            # set the output dirpath from the trainer log dir
            log_dir = Path(trainer.log_dir)
            self.dirpath = str(log_dir / "ckpts")

            # this could be used to add the timestamp to the filename
            # self.timestamp = log_dir.name

        super().setup(trainer=trainer, pl_module=pl_module, stage=stage)
