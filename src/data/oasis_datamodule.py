from pathlib import Path

import torchio

from src.data.base_datamodule import DataModuleBase
from src.data.components.custom_torchio import NDLabelMap, NDScalarImage
from src.data.components.transforms import ClassMapping


class OasisTissueDataModule(DataModuleBase):
    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        dataset_dir: str,
        resampling_target_size: int = 3,
        crop_size: tuple = (64, 64, 64),
        train_val_test_split: list[float] = [0.8, 0.1, 0.1],
        num_workers: int = 0,
        pin_memory: bool | None = False,
    ) -> None:
        """Initialization for the Oasis Tissue Data Module.

        Args:
            batch_size: Batch size used for training.
            input_domain: Target domain of inputs ('kspace' or 'pixel').
            label_domain: Target domain of label ('kspace' or 'pixel').
            dataset_dir: Path to the OASIS dataset.
            train_val_test_split: Split between train, val and test dataset.
                Defaults to [0.8, 0.1, 0.1].
            resampling_target_size: Size to which the input shall be resampled.
                Defaults to 3.
            crop_size: Size to which the input shall be cropped.
                Defaults to (64, 64, 64).
        """
        super().__init__(
            batch_size=batch_size,
            input_domain=input_domain,
            label_domain=label_domain,
            num_classes=7,
            resampling_target_size=resampling_target_size,
            crop_size=crop_size,
            train_val_test_split=train_val_test_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            class_weights=[0.15, 74.13, 5.27, 6.0, 43.65, 121.28, 19.23],
        )
        self.dataset_dir = Path(dataset_dir)

    @property
    def name(self) -> str:
        """Data module name property.

        Returns:
            Data module name.
        """
        return 'OasisTissue'

    def prepare_data(self) -> None:
        """Creates the subject list based on the Oasis dataset dir."""
        self.subject_list = []

        # Iterate over each directory inside the root directory
        for subject_dir in self.dataset_dir.iterdir():
            if subject_dir.name.startswith('OAS1_') and subject_dir.is_dir():
                # Build the mri directory path
                mri_dir_path = subject_dir / 'mri'

                # Check if both files exist in the mri directory
                aseg_path = mri_dir_path / 'aseg.nii.gz'
                brain_mask_path = mri_dir_path / 'brainmask.nii.gz'

                if aseg_path.exists() and brain_mask_path.exists():
                    subject = torchio.Subject(
                        input=NDScalarImage(brain_mask_path),
                        label=NDLabelMap(aseg_path),
                    )
                    self.subject_list.append(subject)

    def get_preprocessing_transform(self) -> torchio.Compose:
        """Composes the transformations for the preprocessing step.

        Returns:
            Transformations for the preprocessing step.
        """
        preprocess = torchio.Compose(
            [
                # Map FreeSurfer label classes to custom label classes
                ClassMapping(),
                torchio.ToCanonical(),
                torchio.Resample('input'),
                torchio.Resample(self.hparams.resampling_target_size),
                torchio.CropOrPad(self.hparams.crop_size),
                torchio.transforms.ZNormalization(),
            ]
        )
        return preprocess


if __name__ == "__main__":
    _ = OasisTissueDataModule()
