import numpy as np
from scipy.signal import fftconvolve
from .utils import padded_cutout_with_center


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

        # Cut/Pad the covariance to the full autocorrelation size
        covariance_size = 2 * cutout_size - 1
        self.noise_covariance, _ = padded_cutout_with_center(
            noise_covariance_full,
            cx,
            cy,
            covariance_size,
        )

        # Apply the conversion factor
        self.noise_covariance *= self.image_conversion_factor ** 2

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

        # Calculate autocorrelation
        autocorrelation = fftconvolve(
            img,
            img[::-1, ::-1],
            mode="full",
        )

        # Normalize the autocorrelation by the number of pairs
        overlap = fftconvolve(
            np.ones_like(img),
            np.ones_like(img)[::-1, ::-1],
            mode="full",
        )
        autocorrelation /= overlap

        return autocorrelation

    def calc_error(
            self,
            weight: np.ndarray,
            xc: int,
            yc: int,
            cutout_size: int,
    ) -> np.ndarray:
        """
        Calculate the error on the aperture flux.

        Parameters
        ----------
        weight: np.ndarray
            Weight function used to calculate the aperture flux
        xc: int
            x coordinate of the center of the aperture
        yc: int
            y coordinate of the center of the aperture
        cutout_size: int
            Size of the cutout

        Returns
        -------
        np.ndarray:
            Errors on the aperture flux
        """

        # Check if there is an RMS map
        if self.rms is None:
            # Calculate the error using the noise image
            return self.background_error(weight)

        else:
            # Calculate the error using the RMS map
            return self.rms_error(weight, xc, yc, cutout_size)

    def background_error(self, weight: np.ndarray) -> np.ndarray:
        """
        Calculate the error on the aperture flux using the noise image.

        Parameters
        ----------
        weight: np.ndarray
            Weight function used to calculate the aperture flux

        Returns
        -------
        np.ndarray:
            Errors on the aperture flux
        """

        # Simplify calculation when uncorrelated noise is assumed
        if self.uncorrelated:
            # Assuming that the negative pixels are just noise
            negative_pixels = self.noise[self.noise < 0]

            # Find the variance of the noise assuming that the noise is centered at 0
            background_variance = (
                                          np.sum(negative_pixels ** 2) / len(negative_pixels)
                                  ) * self.image_conversion_factor ** 2

            # Return the variance using the uncorrelated noise assumption
            return background_variance * np.sum(weight ** 2)

        else:
            # Calculate the autocorrelation of the weight function
            autocorr_weight = fftconvolve(
                weight,
                weight[::-1, ::-1],
                mode="full",
            )

            # Return the variance using the correlated noise
            return np.sum(self.noise_covariance * autocorr_weight)

    def rms_error(
            self,
            weight,
            xc,
            yc,
            cutout_size,
    ) -> np.ndarray:
        """
        Calculate the error on the aperture flux using the noise image and RMS map.

        Parameters
        ----------
        weight: np.ndarray
            Weight function used to calculate the aperture flux

        Returns
        -------
        np.ndarray:
            Errors on the aperture flux
        """

        # Set the correlation kernel if it is not set yet
        if self.kernel is None:
            # Finding the kernel using the local covariance matrix
            cy, cx = np.array(self.noise_covariance.shape) // 2
            self.kernel = (
                    self.noise_covariance
                    / self.noise_covariance[cy, cx]
            )

        # Calculate the RMS prime (weight * RMS)
        rms_cutout, _ = padded_cutout_with_center(
            self.rms,
            xc,
            yc,
            cutout_size,
        )
        rms_prime = (
                rms_cutout
                * weight
                * self.rms_conversion_factor
        )

        # Simplify calculation when uncorrelated noise is assumed
        if self.uncorrelated:
            # Return the variance using the uncorrelated noise assumption
            return np.sum(rms_prime ** 2)

        else:
            # Calculate the autocorrelation of the RMS prime
            autocorr_rms = fftconvolve(
                rms_prime,
                rms_prime[::-1, ::-1],
                mode="full",
            )

            # Return the variance using the correlated noise
            return np.sum(autocorr_rms * self.kernel)
