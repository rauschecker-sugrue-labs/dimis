import torch
import torchio
from einops import pack, parse_shape, rearrange

from src.data.components.custom_torchio import NDTransform


class KSpace(NDTransform, torchio.FourierTransform):
    def __init__(self, exclude_label: bool = False) -> None:
        """Initialization of the KSpace transformation.

        Args:
            exclude_label: Whether to exlcude the label for the transformation.
                Defaults to False.
        """
        super().__init__(exclude_label)

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Applies the 2D real FFT on the given subject.

        Args:
            subject: Subject to be transformed. The data must be in RAS+
                orientation (c, x, y, z).

        Returns:
            Transformed subject.
        """
        for image in self.get_images(subject):
            transformed = torch.fft.rfft2(image.data)
            image.set_data(transformed)
        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the KSpace transformation is invertible.

        Returns:
            Whether the KSpace transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the KSpace transformation.

        Returns:
            Inverse KSpace transformation.
        """
        return InverseKSpace(exclude_label=self.exclude_label)


class InverseKSpace(NDTransform, torchio.FourierTransform):
    def __init__(self, exclude_label: bool = False) -> None:
        """Initialization of the inverse KSpace Transformation.

        Args:
            exclude_label: Whether to exlcude the label for the transformation.
        """
        super().__init__(exclude_label)

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Applies the inverse 2D real FFT on the given subject.

        Args:
            subject: Subject to be transformed. The data must be in RAS+
                orientation (c, x, y, z).

        Returns:
            2D real FFT transformed subject.
        """
        for image in self.get_images(subject):
            inverse_transformed = self._inverse_transform(image)
            image.set_data(inverse_transformed)
        return subject

    @staticmethod
    def _inverse_transform(image: torchio.Image) -> torchio.Image:
        """Inverse 2D real FFT transformation.

        Args:
            image: Image to be transformed. Image data must be in RAS+
                orientation (c, v, x, y, z).

        Returns:
            Transformed image.
        """
        return torch.fft.irfft2(image.data)

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the inverse KSpace transformation is invertible.

        Returns:
            Whether the invserse KSpace transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the inverse KSpace
            transformation.

        Returns:
            KSpace transformation.
        """
        return KSpace(exclude_label=self.exclude_label)


class Complex2Vec(NDTransform, torchio.SpatialTransform):
    def __init__(self, exclude_label: bool = False) -> None:
        """Initialization for the complex to vector transformation."""
        super().__init__(exclude_label)

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Transforms complex numbers to a vector on the given subject.

        Args:
            subject: Subject which contains complex values.

        Returns:
            Subject with at a new dimension for storing the real and imaginary
                part of the image. In case there are not complex numbers, this
                dimension has a length of 1.
        """
        for image in self.get_images(subject):
            if 'complex' in str(image.data.dtype):
                image.set_data(
                    pack([image.data.real, image.data.imag], 'c * x y z')[0]
                )
            else:
                image.set_data(rearrange(image.data, 'c x y z -> c 1 x y z'))

        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the Complex2Vec transformation is invertible.

        Returns:
            Whether the Complex2Vec transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the Complex2Vec transformation.

        Returns:
            Vec2Complex transformation.
        """
        return Vec2Complex()


class Vec2Complex(NDTransform, torchio.SpatialTransform):
    def __init__(self, exclude_label: bool = False) -> None:
        """Initialization for the vector to complex transformation."""
        super().__init__(exclude_label)

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Transforms vectors to complex numbers on the given subject.

        Args:
            subject: Subject to be transformed. Image data must be in RAS+
                orientation (c, v, x, y, z) where the channel dim contains
                seg classes and vector dim the real and imaginary values of a
                complex number.

        Returns:
            Subject with complex numbers.
        """
        for image in self.get_images(subject):
            image.set_data(self._inverse_transform(image.data))
        return subject

    @staticmethod
    def _inverse_transform(data: torchio.Image) -> torchio.Image:
        """Inverse Vec2Complex transformation.

        Args:
            image: Image to be transformed. Image data must be in RAS+
                orientation (c, v, x, y, z).

        Returns:
            Transformed image.
        """
        # Reorder the axis since view_as_complex expects the last dimension
        # represents the real and imaginary components of complex numbers.
        data = rearrange(data, 'c v x y z -> c x y z v').contiguous()
        return torch.unsqueeze(torch.view_as_complex(data), dim=1)

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the Vec2Complex transformation is invertible.

        Returns:
            Whether the Vec2Complex transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the Vec2Complex transformation.

        Returns:
            Complex2Vec transformation.
        """
        return Complex2Vec()


class Unsqueeze(NDTransform, torchio.SpatialTransform):
    def __init__(self, position: int) -> None:
        """Initialization for the unsqueeze transformation.

        Args:
            position: Position at which the tensor should be unsqueezed.
        """
        super().__init__()
        self.position = position

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Transformation that unsqueezes the tensor at the specified position.

        Args:
            subject: Subject containing data to be unsqueezed.

        Returns:
            Subject with one additional dimension at the specified position.
        """
        for image in self.get_images(subject):
            image.set_data(torch.unsqueeze(image.data, dim=self.position))
        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the unsqueeze transformation is invertible.

        Returns:
            Whether the unsqueeze transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the unsqueeze
            transformation.

        Returns:
            Squeeze transformation.
        """
        return Squeeze()


class Squeeze(NDTransform, torchio.SpatialTransform):
    def __init__(self, position: int) -> None:
        """Initialization for the squeeze transformation.

        Args:
            position: Position at which the tensor should be squeezed.
        """
        super().__init__(exclude='input')
        self.position = position

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Transformation that squeezes the tensor.

        Args:
            subject: Subject containing data to be squeezed.

        Returns:
            Subject with one dimension less at the specified position.
        """
        for image in self.get_images(subject):
            image.set_data(torch.squeeze(image.data, dim=self.position))
        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the squeeze transformation is invertible.

        Returns:
            Whether the squeeze transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the squeeze
            transformation.

        Returns:
            Unsqueeze transformation.
        """
        return Unsqueeze()


class Compress(NDTransform, torchio.SpatialTransform):
    def __init__(self, exclude_label: bool = False) -> None:
        """Initialization for compression transformation.
        Args:
            exclude_label: Whether to exlcude the label for the transformation.
        """
        super().__init__(exclude_label)

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Performs lossless compression on the vecotrized 2D rfft.
        Apply this transform after :class:`Complex2Vec` which itself
        has been applied after py:class:: KSpace.

        If the original image has size (1, 1, 64, 64, 64), after applying
        :class:`KSpace`, we get a (1, 1, 64, 64, 33) complex tensor, and
        after applying :class:`Complex2Vec`, we get a (1, 2, 64, 64, 33)
        real tensor. The dimension of this tensor is larger than the original
        image. However, there is some redundancy here. The Compress transform
        applies a lossless compression using conjugate symmetries present in
        the first and last columns of the tensor to reduce its dimensions to
        (1, 2, 64, 64, 32) which is equivalent to the original image.

        Args:
            subject: Subject which vectorized KSpace data.

        Returns:
            Subject with compressed data.
        """
        for image in self.get_images(subject):
            data = image.data
            shape = parse_shape(data, '_ _ _ y z')
            real = pack(
                [
                    data[:, 0, :, : shape['y'] // 2 + 1, 0:1],
                    data[:, 1, :, 1 : (shape['y'] + 1) // 2, 0:1],
                ],
                'b x * z',
            )[0]
            imag = pack(
                [
                    data[
                        :,
                        0,
                        :,
                        : shape['y'] // 2 + 1,
                        shape['z'] - 1 : shape['z'],
                    ],
                    data[
                        :,
                        1,
                        :,
                        1 : (shape['y'] + 1) // 2,
                        shape['z'] - 1 : shape['z'],
                    ],
                ],
                'b x * z',
            )[0]

            compressed_col = rearrange([real, imag], 'v c x y z -> c v x y z')
            remaining_cols = data[:, :, :, :, 1:-1]
            image.set_data(
                pack([compressed_col, remaining_cols], 'c v x y *')[0]
            )

        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the compress transformation is invertible.

        Returns:
            Whether the compress transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the compress
            transformation.

        Returns:
            Decompress transformation.
        """
        return Decompress()


class Decompress(NDTransform, torchio.SpatialTransform):
    def __init__(self, exclude_label: bool = False) -> None:
        """Initialization for decompression transformation."""
        super().__init__(exclude_label)

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Performs decompression on tensor that are compressed by
         :class:`Compress`.

        Args:
            subject: Subject compressed data.

        Returns:
            Subject with decompressed data.
        """
        for image in self.get_images(subject):
            data = image.data
            shape = parse_shape(data, '_ _ _ y _')

            real = pack(
                [
                    data[:, :, :, 0:1, 0:1],
                    data[:, :, :, 1 : (shape['y'] + 1) // 2, 0:1],
                    data[
                        :,
                        :,
                        :,
                        (shape['y'] + 1) // 2 : shape['y'] // 2 + 1,
                        0:1,
                    ],
                    torch.flip(
                        data[:, :, :, 1 : (shape['y'] + 1) // 2, 0:1],
                        dims=(-1,),
                    ),
                ],
                'c z x * v',
            )[0]

            imag = torch.zeros_like(real)
            imag[:, :, :, 1 : (shape['y'] + 1) // 2, 0:1] = data[
                :, :, :, shape['y'] // 2 + 1 :, 0:1
            ]
            imag[:, :, :, shape['y'] // 2 + 1 :, 0:1] = -torch.flip(
                data[:, :, :, shape['y'] // 2 + 1 :, 0:1], dims=(-1,)
            )

            cols_decompressed = rearrange(
                pack([real, imag], 'c z x y *')[0], 'c z x y v -> c v x y z'
            )

            data = pack(
                [
                    cols_decompressed[:, :, :, :, 0:1],
                    data[:, :, :, :, 1:],
                    cols_decompressed[:, :, :, :, 1:],
                ],
                'c v x y *',
            )[0]

            image.set_data(data)

        return subject

    @staticmethod
    def is_invertible() -> bool:
        """Shows whether the decompress transformation is invertible.

        Returns:
            Whether the decompress transformation is invertible.
        """
        return True

    def inverse(self):
        """Gives the inverse transformation for the decompress
            transformation.

        Returns:
            Compress transformation.
        """
        return Compress()


class ClassMapping(NDTransform):
    def __init__(self) -> None:
        """Initialization for class mapping transformation."""
        super().__init__(include=['label'])

        # Map FreeSurfer label values to kseg tissue label values
        self.mapping = {
            0: 0,
            2: 3,
            3: 2,
            4: 1,
            5: 1,
            7: 6,
            8: 6,
            10: 4,
            11: 4,
            12: 4,
            13: 4,
            14: 1,
            15: 1,
            16: 5,
            17: 4,
            18: 4,
            24: 1,
            26: 4,
            28: 4,
            30: 1,
            41: 3,
            42: 2,
            43: 1,
            44: 1,
            46: 6,
            47: 6,
            49: 4,
            50: 4,
            51: 4,
            52: 4,
            53: 4,
            54: 4,
            58: 4,
            60: 4,
            62: 1,
            72: 1,
            78: 3,
            79: 3,
            81: 4,
            82: 4,
            85: 5,
        }

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Performs segmentation class mapping on tensor.

        Args:
            subject: Subject data.

        Returns:
            Subject with new segmentation classes.
        """
        for image in self.get_images(subject):
            data = image.data
            # This is used to prevent replacing replaced values
            replaced_mask = torch.zeros_like(data).bool()

            for original, new in self.mapping.items():
                mask = (data == original) & (~replaced_mask)
                data[mask] = new
                replaced_mask[mask] = True
            image.set_data(data)
        return subject
