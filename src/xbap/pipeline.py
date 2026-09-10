from .flux import flux
from .noise import NoiseModel
from .psf import PSFDeconvolver
import numpy as np


def xbap_flux(image: np.ndarray,
              psf: np.ndarray | str,
              centers: np.ndarray,
              weight_sizes: float | np.ndarray,
              noise: np.ndarray | None = None,
              rms: np.ndarray | None = None,
              cutout_size: int = 128,
              image_conversion_factor: float = 1.0,
              rms_conversion_factor: float = 1.0,
              show_progress: bool = False,
              uncorrelated: bool = False,
              eps: float = 1e-8
              ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate the X-BAP aperture flux.

    Parameters
    ----------
    image: np.ndarray
        Image in which the aperture flux will be calculated.
    psf: np.ndarray | str
        PSF of the image or directory to .psf file from PSFex.
    centers: np.ndarray
        Centers of the Gaussian apertures.
    weight_sizes: float | np.ndarray, shape (N,) or (N, 3)
        Parameters determining the shape of the apertures, with supported shapes.

        - scalar: same scale circular Gaussian apertures is used for all centers.

        - (3,): elliptical Gaussian aperture is defined by three parameters:
        the scale parameter along the x-axis, the scale parameter along the y-axis,
        and the rotation angle relative to the vertical axis. The same elliptical
        Gaussian aperture is used for all sources

        - (N,): one cicular Gaussian aperture per center.

        - (N, 3): elliptical Gaussian aperture is defined by three parameters:
        the scale parameter along the x-axis, the scale parameter along the y-axis,
        and the rotation angle relative to the vertical axis. One elliptical
        Gaussian aperture is used for each source.
    noise: np.ndarray | None = None
        Noise in the image..
    rms: np.ndarray | None = None
        RMS map of the noise in the image.
    cutout_size: int = 128
        Size of the cutout.
    image_conversion_factor: float = 1.0
        Conversion factor.
    rms_conversion_factor: float = 1.0
        Conversion factor.
    show_progress: bool = False
        Whether to show the progress.
    uncorrelated: bool = False
        Whether to assume uncorrelated noise in the image.
    eps: float = 1e-8
        Small value to avoid division by zero in the prefactor of the deconvolution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray|None]
        The calculated X-BAP aperture flux. When measurement errors are included,
        returns (flux, error) else (flux, None).

    """

    if noise is None and rms is not None:
        raise ValueError("If RMS map is provided, noise must also be provided.")

    # Initialize PSF deconvolver
    psf_deconvolver = PSFDeconvolver(psf, eps)
    psf_deconvolver.prepare(cutout_size)

    # Initialize noise model
    if noise is not None:
        noise_model = NoiseModel(
            noise=noise,
            rms=rms,
            image_conversion_factor=image_conversion_factor,
            rms_conversion_factor=rms_conversion_factor,
            uncorrelated=uncorrelated
        )
        noise_model.set_noise_covariance(cutout_size)
    else: 
        noise_model = None

    # Calculate flux (and optionally error)
    return flux(image,
                centers,
                weight_sizes,
                psf_deconvolver,
                noise_model,
                cutout_size=cutout_size,
                image_conversion_factor=image_conversion_factor,
                show_progress=show_progress)
