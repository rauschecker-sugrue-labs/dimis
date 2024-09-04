import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vfunctional
import torchvision.utils as vutils
from einops import rearrange
from lightning import LightningModule
from torchmetrics import Dice, MaxMetric, MeanMetric, Recall, Specificity

from src.data.components.transforms import (
    Decompress,
    InverseKSpace,
    Vec2Complex,
)
from src.utils.math import min_max_scale


class DIMISLitModule(LightningModule):
    """The `LightningModule` for DIMIS.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.Module,
        num_classes: int,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        Args:
            net: The network to train.
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler to use for training.
            criterion: The criterion to use for calculating the loss.
            num_classes: Number of segmentation classes.
            compile: Whether to convert code into optimized TorchScript graphs.
        """
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = criterion

        # Metric objects for calculating and averaging accuracy across batches
        self.train_dice = Dice(num_classes=num_classes, average='macro')
        self.val_dice = Dice(num_classes=num_classes, average='macro')
        self.test_dice = Dice(num_classes=num_classes, average='macro')
        self.val_recall = Recall(
            task='multiclass', num_classes=num_classes, average='macro'
        )
        self.test_recall = Recall(
            task='multiclass', num_classes=num_classes, average='macro'
        )
        self.val_specificity = Specificity(
            task='multiclass', num_classes=num_classes, average='macro'
        )
        self.test_specificity = Specificity(
            task='multiclass', num_classes=num_classes, average='macro'
        )

        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # For tracking best so far validation accuracy
        self.val_dice_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        Args:
            x: A tensor of images.

        Returns:
            A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_dice.reset()
        self.val_dice_best.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        Returns:
            A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
        """
        x, y = batch['input'].float(), batch['label'].float()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss, logits

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx: The index of the current batch.

        Returns:
            A tensor of losses between model predictions and targets.
        """
        loss, logits = self.model_step(batch)
        x, y = batch['input'], batch['label']

        x, y, logits = self._to_spatial(x, y, logits)

        # Convert logits to predicted segmentation
        preds = torch.argmax(logits, dim=1)
        preds = rearrange(
            F.one_hot(preds, num_classes=logits.shape[1]),
            'b v x y z c -> b c v x y z',
        )

        # Scale gt to [0,1] which is required for some metrics
        y = min_max_scale(y)

        # Parse tensor to closes int values which is required for some metrics
        y = y.round().int()

        # Update and log metrics
        self.train_loss(loss)
        self.train_dice(preds, y)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/dice",
            self.train_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx: The index of the current batch.
        """
        loss, logits = self.model_step(batch)
        x, y = batch['input'], batch['label']

        x, y, logits = self._to_spatial(x, y, logits)

        # Convert logits to predicted segmentation
        preds = torch.argmax(logits, dim=1)
        preds = rearrange(
            F.one_hot(preds, num_classes=logits.shape[1]),
            'b v x y z c -> b c v x y z',
        )

        # Scale gt to [0,1] which is required for some metrics
        y = min_max_scale(y)

        # Parse tensor to closes int values which is required for some metrics
        y = y.round().int()

        # Update and log metrics
        self.val_loss(loss)
        self.val_dice(preds, y)
        self.val_specificity(preds, y)
        self.val_recall(preds, y)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/dice",
            self.val_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/specificity",
            self.val_specificity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Log qualitative results
        if batch_idx == 0:
            self.log_qualitativ_results(
                x, y, preds, self.logger, batch_idx, self.current_epoch
            )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        dice = self.val_dice.compute()  # get current val acc
        self.val_dice_best(dice)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/dice_best",
            self.val_dice_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_dice(preds, targets)
        self.test_specificity(preds, targets)
        self.test_recall(preds, targets)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/dice",
            self.test_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/specificity",
            self.test_specificity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/recall",
            self.test_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Args:
            stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters()
        )
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def _to_spatial(
        self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor
    ) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transforms network input, ground truth and logits into spatial domain.

        Args:
            x: Network input.
            y: Ground truth.
            logits: Raw output of the network.

        Returns:
            Variables in spatial domain.
        """
        vec2complex = Vec2Complex()
        decompress = Decompress()
        inv_kspace = InverseKSpace()
        transformed_variables = []

        for variable in (x, y, logits):
            # If channel dimension contains real and imag part, transform.
            if variable.shape[2] == 2:
                variable = [
                    inv_kspace(vec2complex(decompress(b)))
                    for b in torch.unbind(variable, dim=0)
                ]
                variable = torch.stack(variable, dim=0)
            transformed_variables.append(variable)
        return transformed_variables

    def log_qualitativ_results(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        logger: any,
        batch_idx: int,
        epoch: int,
        stage: str = 'val',
    ) -> None:
        """Logs the qualitative results of the segmentation prediction.

        Args:
            x: Input.
            y: Ground truth.
            y_hat: Prediction.
            logger: Logger which is used for logging the results.
            batch_idx: Current batch index.
            epoch: Current epoch.
            stage: Current stage. Defaults to 'val'.
        """
        # Extract middle slice along the z-axis
        middle_z = x.shape[-1] // 2
        x_slice = x[batch_idx, 0, 0, :, :, middle_z].cpu().numpy()
        y_slice = y[batch_idx, :, 0, :, :, middle_z].cpu().numpy()
        y_hat_slice = y_hat[batch_idx, :, 0, :, :, middle_z].cpu().numpy()

        # Convert one-hot to ordinal encoded ground truth and prediction
        y_slice = np.argmax(y_slice, axis=0)
        y_hat_slice = np.argmax(y_hat_slice, axis=0)

        # Define a colormap where each class ID maps to an RGB color
        color_map = {
            0: [0, 0, 0],  # Black for class 0 (background)
            1: [0, 255, 0],  # Green for class 1 (CSF / femoral cartilage)
            2: [255, 0, 0],  # Red for class 2 (cortical GM / tibial cartilage)
            3: [0, 0, 255],  # Blue for class 3 (WM / patellar cartilage)
            4: [255, 255, 0],  # Yellow for class 4 (GM / femur)
            5: [0, 255, 255],  # Cyan for class 5 (brain stem / tibia)
            6: [255, 0, 255],  # Magenta for class 6 (cerebellum / patella)
        }

        # Transform x
        x_slice = cv2.normalize(
            x_slice, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U
        )
        x_slice = cv2.cvtColor(x_slice, cv2.COLOR_GRAY2RGB)
        x_slice = np.transpose(torch.from_numpy(x_slice), axes=[2, 0, 1])
        x_slice = vfunctional.rotate(x_slice, 90)

        # Transform y
        output_image = np.zeros(
            (y_slice.shape[0], y_slice.shape[1], 3), dtype=np.uint8
        )
        for value, color in color_map.items():
            mask = y_slice == value
            output_image[mask] = color
        y_slice = np.transpose(torch.from_numpy(output_image), axes=[2, 0, 1])
        y_slice = vfunctional.rotate(y_slice, 90)
        y_slice = torch.from_numpy(
            cv2.addWeighted(x_slice.numpy(), 0.5, y_slice.numpy(), 0.5, 0)
        )

        # Transform y_hat
        output_image = np.zeros(
            (y_hat_slice.shape[0], y_hat_slice.shape[1], 3), dtype=np.uint8
        )
        for value, color in color_map.items():
            mask = y_hat_slice == value
            output_image[mask] = color
        y_hat_slice = np.transpose(
            torch.from_numpy(output_image), axes=[2, 0, 1]
        )
        y_hat_slice = vfunctional.rotate(y_hat_slice, 90)
        y_hat_slice = torch.from_numpy(
            cv2.addWeighted(x_slice.numpy(), 0.5, y_hat_slice.numpy(), 0.5, 0)
        )

        # Create grid and log to TensorBoard
        seg_grid = vutils.make_grid([x_slice, y_slice, y_hat_slice])
        logger.experiment.add_image(
            f'{stage}_sample_{batch_idx}', seg_grid, float(epoch)
        )


if __name__ == "__main__":
    _ = DIMISLitModule(None, None, None, None)
