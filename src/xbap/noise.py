import numpy as np
from scipy.signal import fftconvolve
from .utils import padded_cutout_with_center, find_best_square_coords


class NoiseModel:
    def __init__(
        self,
        noise: np.ndarray | None = None,
        rms: np.ndarray | None = None,
        image_conversion_factor: float = 1.0,
        rms_conversion_factor: float = 1.0,
        uncorrelated: bool = False,
    ) -> None:

        self.noise = noise
        self.rms = rms
        self.image_conversion_factor = image_conversion_factor
        self.rms_conversion_factor = rms_conversion_factor
        self.uncorrelated = uncorrelated

        self.noise_covariance = None
        self.kernel = None

    def set_noise_covariance(self, cutout_size: int) -> None:
        """
        Sets the noise covariance matrix from the noise square
        which is used for the error calculation.

        Parameters
        ----------
        cutout_size: int
            Size of the cutout.
        """

        # Calculate the local covariance matrix from the noise
        noise_covariance_full = self._covariance_fft2d(self.noise)

        cy, cx = np.array(noise_covariance_full.shape) // 2

        # Cut/Pad the covariance to the cutout size
        self.noise_covariance, _ = padded_cutout_with_center(
            noise_covariance_full,
            cx,
            cy,
            cutout_size,
        )

        # Apply the conversion factor
        self.noise_covariance *= self.image_conversion_factor**2

    def _covariance_fft2d(self, noise_image: np.ndarray) -> np.ndarray:
        """
        Calculate the local covariance matrix from the noise square

        Parameters
        ----------
        noise_image: np.ndarray
            Noise image used to estimate the local covariance matrix

        Returns
        -------
        np.ndarray:
            Local covariance matrix
        """

        # Store noise image dimensions
        height, width = noise_image.shape

        # Remove mean
        img = noise_image.copy()
        img -= np.mean(img)

        # Calculate autocorrelation and normalize the result
        autocorrelation = fftconvolve(img, img[::-1, ::-1], mode="same")

        overlap = fftconvolve(
            np.ones_like(img),
            np.ones_like(img)[::-1, ::-1],
            mode="same",
        )

        autocorrelation /= overlap

        return autocorrelation

    def calc_error(self, weight: np.ndarray, xc: int, yc: int, cutout_size: int) -> np.ndarray:
        if self.rms is None:
            return self.background_error(weight)
        else:
            return self.rms_error(weight, xc, yc, cutout_size)

    def background_error(self, weight: np.ndarray) -> np.ndarray:
        if self.uncorrelated:
            negative_pixels = self.noise[self.noise < 0]

            background_variance = (
                np.sum(negative_pixels**2) / len(negative_pixels)
            ) * self.image_conversion_factor**2
            return background_variance * np.sum(weight**2)

        autocorr_weight = fftconvolve(weight, weight[::-1, ::-1], mode="same")

        return np.sum(self.noise_covariance * autocorr_weight)

    def rms_error(self, weight, xc, yc, cutout_size) -> np.ndarray:
        if self.kernel is None:
            cy, cx = np.array(self.noise_covariance.shape) // 2
            self.kernel = self.noise_covariance / self.noise_covariance[cy, cx]

        rms_cutout, _ = padded_cutout_with_center(self.rms, xc, yc, cutout_size)
        weight_prime = rms_cutout * weight * self.rms_conversion_factor
        if self.uncorrelated:
            return np.sum(weight_prime ** 2)
        conv = fftconvolve(weight_prime, self.kernel, mode="same")
        return np.sum(weight_prime * conv)
