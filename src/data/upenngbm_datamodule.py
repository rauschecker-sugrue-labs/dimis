import glob
import os

import torch
import torchio

from src.data.base_datamodule import DataModuleBase
from src.data.components.custom_torchio import NDLabelMap, NDScalarImage


class UPennGBMSSDataModule(DataModuleBase):
    def __init__(
        self,
        batch_size: int,
        input_domain: str,
        label_domain: str,
        datasets_dir: str,
        train_val_ratio: float = 0.8,
        resampling_target_size: int = 3,
        crop_size: tuple = (64, 64, 64),
    ) -> None:
        """Initialization for the UPenn GBM skull stripping Data Module.

        Args:
            batch_size: Batch size used for training.
            input_domain: Target domain of inputs ('kspace' or 'pixel').
            label_domain: Target domain of label ('kspace' or 'pixel').
            datasets_dir: Path to the datasets directory.
            train_val_ratio: Ratio between train dataset and val+test dataset.
                Defaults to 0.8.
            resampling_target_size: Size to which the input shall be resampled.
                Defaults to 3.
            crop_size: Size to which the input shall be cropped.
                Defaults to (64, 64, 64).
        """
        super().__init__(
            batch_size=batch_size,
            input_domain=input_domain,
            label_domain=label_domain,
            dataset_dir=os.path.join(os.getcwd(), datasets_dir, 'UPENN_GBM/'),
            num_classes=2,
            train_val_ratio=train_val_ratio,
            resampling_target_size=resampling_target_size,
            crop_size=crop_size,
        )

    @property
    def name(self) -> str:
        """Data module name property.

        Returns:
            Data module name.
        """
        return 'UPennGBMSS'

    def prepare_data(self) -> None:
        """Creates the subject list based on the UPennGBM dataset dir."""
        self.subject_list = []
        image_dir = os.path.join(
            self.dataset_dir, 'images_structural_unstripped'
        )
        stripped_dir = os.path.join(self.dataset_dir, 'images_structural')

        # Go through each subject's folder
        for subject_dir in glob.glob(os.path.join(image_dir, '*')):
            # Skip the directories ending with 21
            if subject_dir.endswith('21'):
                continue

            # Find the unstripped T1 image path
            t1_unstripped_path = glob.glob(
                os.path.join(subject_dir, '*_T1_unstripped.nii.gz')
            )[0]
            base_name = os.path.basename(t1_unstripped_path).replace(
                '_T1_unstripped.nii.gz', ''
            )

            # Determine the corresponding stripped T1 image path
            t1_stripped_path = os.path.join(
                stripped_dir, base_name, f'{base_name}_T1.nii.gz'
            )

            # Create and append subject object to list
            subject = torchio.Subject(
                input=NDScalarImage(t1_unstripped_path),
                label=NDLabelMap(t1_stripped_path),
            )
            self.subject_list.append(subject)

    def get_preprocessing_transform(self) -> torchio.Compose:
        """Composes the transformations for the preprocessing step.

        Returns:
            Transformations for the preprocessing step.
        """
        preprocess = torchio.Compose(
            [
                # Convert stripped images to binary segmentation maps
                torchio.Lambda(
                    lambda x: (x != 0).float(), types_to_apply=[torchio.LABEL]
                ),
                torchio.ToCanonical(),
                torchio.Resample('input'),
                torchio.Resample(self.resampling_target_size),
                torchio.CropOrPad(self.crop_size),
                torchio.transforms.ZNormalization(),
            ]
        )
        return preprocess

    class UPennGBMTumorDataModule(DataModuleBase):
        def __init__(
            self,
            batch_size: int,
            input_domain: str,
            label_domain: str,
            datasets_dir: str,
            train_val_ratio: float = 0.8,
            resampling_target_size: int = 3,
            crop_size: tuple = (64, 64, 64),
        ) -> None:
            """Initialization for the UPenn GBM tumor Data Module.

            Args:
                batch_size: Batch size used for training.
                input_domain: Target domain of inputs ('kspace' or 'pixel').
                label_domain: Target domain of label ('kspace' or 'pixel').
                datasets_dir: Path to the datasets directory.
                train_val_ratio: Ratio between train dataset and val+test dataset.
                    Defaults to 0.8.
                resampling_target_size: Size to which the input shall be resampled.
                    Defaults to 3.
                crop_size: Size to which the input shall be cropped.
                    Defaults to (64, 64, 64).
            """
            super().__init__(
                batch_size=batch_size,
                input_domain=input_domain,
                label_domain=label_domain,
                dataset_dir=os.path.join(
                    os.getcwd(), datasets_dir, 'UPENN_GBM/'
                ),
                num_classes=4,
                train_val_ratio=train_val_ratio,
                resampling_target_size=resampling_target_size,
                crop_size=crop_size,
                class_weights=[0.25, 196.26, 42.66, 114.3],
            )

        @property
        def name(self) -> str:
            """Data module name property.

            Returns:
                Data module name.
            """
            return 'UPennGBMTumor'

        def prepare_data(self) -> None:
            """Creates the subject list based on the UPennGBM dataset dir."""
            self.subject_list = []
            image_paths = []
            segmentation_paths = []
            image_dir = os.path.join(self.dataset_dir, 'images_structural')
            manual_segm_dir = os.path.join(self.dataset_dir, 'images_segm')
            auto_segm_dir = os.path.join(self.dataset_dir, 'automated_segm')

            # Go through each subject's folder
            for subject_dir in glob.glob(os.path.join(image_dir, '*')):
                # Skip the directories ending with 21
                if subject_dir.endswith('21'):
                    continue

                # Look for the T1 image
                t1_image_path = glob.glob(
                    os.path.join(subject_dir, '*_T1.nii.gz')
                )[
                    0
                ]  # Taking the first match
                image_paths.append(t1_image_path)
                base_name = os.path.basename(t1_image_path).split('_T1.nii.gz')[
                    0
                ]

                # Check for manually segmented labels
                manual_segm = os.path.join(
                    manual_segm_dir, f'{base_name}_segm.nii.gz'
                )
                if os.path.exists(manual_segm):
                    segmentation_paths.append(manual_segm)
                else:
                    # If no manually segmented label, check for automatically
                    # segmented label
                    auto_segm = os.path.join(
                        auto_segm_dir,
                        f'{base_name}_automated_approx_segm.nii.gz',
                    )
                    segmentation_paths.append(auto_segm)

            # Create subject objects
            for image_path, seg_path in zip(image_paths, segmentation_paths):
                subject = torchio.Subject(
                    input=NDScalarImage(image_path),
                    label=NDLabelMap(seg_path),
                )
                self.subject_list.append(subject)

        def get_preprocessing_transform(self) -> torchio.Compose:
            """Composes the transformations for the preprocessing step.

            Returns:
                Transformations for the preprocessing step.
            """
            preprocess = torchio.Compose(
                [
                    # Change segmentation class 4 to 3 since 3 is never used.
                    torchio.Lambda(
                        lambda x: torch.where(x == 4, torch.tensor(3), x),
                        types_to_apply=[torchio.LABEL],
                    ),
                    torchio.ToCanonical(),
                    torchio.Resample('input'),
                    torchio.Resample(self.resampling_target_size),
                    torchio.CropOrPad(self.crop_size),
                    torchio.transforms.ZNormalization(),
                ]
            )
            return preprocess


if __name__ == "__main__":
    _ = UPennGBMSSDataModule()
