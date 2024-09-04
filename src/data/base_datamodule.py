import torch
import torchio
from lightning import LightningDataModule
from torch.utils.data import Dataset, random_split

from src.data.components.dataloader import SubjectDataLoader
from src.data.components.transforms import (
    Complex2Vec,
    Compress,
    KSpace,
    Unsqueeze,
)


class DataModuleBase(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.
    """

    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        num_classes: int,
        resampling_target_size: int,
        crop_size: tuple,
        train_val_test_split: list[float] = [0.8, 0.1, 0.1],
        num_workers: int = 0,
        pin_memory: bool | None = False,
        class_weights: list | None = None,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(0.8, 0.1, 0.1)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes.
        """
        return self.hparams.num_classes

    @property
    def input_shape(self) -> tuple[int]:
        """Input shape property.

        Returns:
            Input shape.
        """
        if self.hparams.input_domain == 'kspace':
            return (
                1,
                2,
                self.hparams.crop_size[0],
                self.hparams.crop_size[1],
                self.hparams.crop_size[2] // 2,
            )
        return (
            1,
            1,
            self.hparams.crop_size[0],
            self.hparams.crop_size[1],
            self.hparams.crop_size[2],
        )

    @property
    def label_shape(self) -> tuple[int]:
        """Label shape property.

        Returns:
            Label shape.
        """
        if self.hparams.label_domain == 'kspace':
            return (
                self.hparams.num_classes,
                2,
                self.hparams.crop_size[0],
                self.hparams.crop_size[1],
                self.hparams.crop_size[2] // 2,
            )
        return (
            self.hparams.num_classes,
            1,
            self.hparams.crop_size[0],
            self.hparams.crop_size[1],
            self.hparams.crop_size[2],
        )

    def get_preprocessing_transform(self) -> torchio.Compose:
        """Composes the transformations for the preprocessing step.

        Returns:
            Transformations for the preprocessing step.
        """
        preprocess = torchio.Compose(
            [
                torchio.ToCanonical(),
                torchio.Resample('input'),
                torchio.Resample(self.resampling_target_size),
                torchio.CropOrPad(self.crop_size),
                torchio.transforms.ZNormalization(),
            ]
        )
        return preprocess

    def get_augmentation_transform(self) -> torchio.Compose:
        """Composes the transformations for the augmentation step.

        Returns:
            Transformations for the augmentation step.
        """
        augment = torchio.Compose(
            [
                torchio.RandomAffine(),
                torchio.RandomGamma(p=0.1),
                torchio.RandomNoise(p=0.1),
                torchio.RandomMotion(p=0.1),
                torchio.RandomBiasField(p=0.1),
            ]
        )
        return augment

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # Split dataset
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=self.subject_list,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )

        preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()

        # If the input domain is not in k-space, neither is the label domain
        if self.hparams.input_domain == 'kspace':
            domain_transform = torchio.Compose(
                [
                    KSpace(
                        exclude_label=(self.hparams.label_domain == 'pixel')
                    ),
                    Complex2Vec(
                        exclude_label=(self.hparams.label_domain == 'pixel')
                    ),
                    Compress(
                        exclude_label=(self.hparams.label_domain == 'pixel')
                    ),
                ]
            )
            self.train_transform = torchio.Compose(
                [
                    preprocess,
                    augment,
                    torchio.OneHot(num_classes=self.hparams.num_classes),
                    domain_transform,
                ]
            )
            self.val_transform = torchio.Compose(
                [
                    preprocess,
                    torchio.OneHot(num_classes=self.hparams.num_classes),
                    domain_transform,
                ]
            )

            self.test_transform = self.val_transform
        else:
            self.train_transform = torchio.Compose(
                [
                    preprocess,
                    augment,
                    torchio.OneHot(num_classes=self.hparams.num_classes),
                    Unsqueeze(position=1),
                ]
            )
            self.val_transform = torchio.Compose(
                [
                    preprocess,
                    torchio.OneHot(num_classes=self.hparams.num_classes),
                    Unsqueeze(position=1),
                ]
            )
            self.test_transform = self.val_transform

        self.train_set = torchio.SubjectsDataset(
            self.data_train, transform=self.train_transform
        )
        self.val_set = torchio.SubjectsDataset(
            self.data_val, transform=self.val_transform
        )
        self.test_set = torchio.SubjectsDataset(
            self.data_test, transform=self.test_transform
        )

    def train_dataloader(self) -> SubjectDataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return SubjectDataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> SubjectDataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return SubjectDataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> SubjectDataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return SubjectDataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> dict[any, any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DataModuleBase()
