from .flux import flux
from .noise import NoiseModel
from .psf import PSFDeconvolver
import numpy as np


def xbap_flux(image: np.ndarray,
              psf: np.ndarray,
              centers: np.ndarray,
              weight_sizes: float | np.ndarray,
              noise: np.ndarray | None = None,
              rms: np.ndarray | None = None,
              cutout_size: int = 128,
              image_conversion_factor: float = 1.0,
              rms_conversion_factor: float = 1.0,
              show_progress: bool = False,
              uncorrelated: bool = False,
              eps: float = 1e-8,
              psf_deconvolver: PSFDeconvolver | None = None,
              noise_model: NoiseModel | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate the X-BAP aperture flux.

    Parameters
    ----------
    image: np.ndarray
        Image in which the aperture flux will be calculated.
    psf: np.ndarray
        PSF of the image.
    centers: np.ndarray
        Centers of the Gaussian apertures.
    weight_sizes: float | np.ndarray, shape (N,) or (N, 3)
        Parameters determining the shape of the apertures, with supported shapes.

        - scalar: same scale circular Gaussian apertures is used for all centers.

        - (N,): one cicular Gaussian aperture per center.

        - (N, 3): each elliptical Gaussian aperture is defined by three parameters:
        the scale parameter along the x-axis, the scale parameter along the y-axis,
        and the rotation angle relative to the vertical axis.
    rms: np.ndarray | None = None
        RMS map of the noise in the image.
    calculate_noise: bool = True
        Whether to calculate the noise.
    cutout_size: int = 128
        Size of the cutout.
    noise_square_size: int = 128
        Size of the noise square.
    image_conversion_factor: float = 1.0
        Conversion factor.
    rms_conversion_factor: float = 1.0
        Conversion factor.
    show_progress: bool = False
        Whether to show the progress.
    uncorrelated: bool = False
        Whether to assume uncorrelated noise in the image.
    psf_deconvolver: PSFDeconvolver
        PSF deconvolution algorithm.
    noise_model: NoiseModel
        Noise model.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        The calculated X-BAP aperture flux. When measurement errors are included,
        returns (flux, error).

    """

    # Initialize PSF deconvolver
    if psf_deconvolver is None:
        psf_deconvolver = PSFDeconvolver(psf)
        psf_deconvolver.prepare(cutout_size, eps)

    # Initialize noise model
    if noise is not None and noise_model is None:
        noise_model = NoiseModel(
            noise=noise,
            rms=rms,
            image_conversion_factor=image_conversion_factor,
            rms_conversion_factor=rms_conversion_factor,
            uncorrelated=uncorrelated
        )
        noise_model.set_noise_covariance(cutout_size)

    # Calculate flux (and optionally error)
    return flux(image,
                centers,
                psf_deconvolver,
                weight_sizes,
                noise_model,
                cutout_size=cutout_size,
                image_conversion_factor=image_conversion_factor,
                show_progress=show_progress)
