from .flux import flux
from .noise import NoiseModel
from .psf import PSFDeconvolver
import numpy as np


def gaap_flux(image: np.ndarray,
              psf: np.ndarray,
              centers: np.ndarray,
              weight_sizes,
              rms: np.ndarray | None = None,
              calculate_noise: bool = True,
              cutout_size: int = 128,
              noise_square_size: int = 128,
              image_conversion_factor: float = 1.0,
              rms_conversion_factor: float = 1.0,
              show_progress: bool = True,
              uncorrelated: bool = False,
              bilinear_interpolation: bool = False,
              psf_deconvolver=None,
              noise_model=None):

    if psf_deconvolver is None:
        psf_deconvolver = PSFDeconvolver(psf)
        psf_deconvolver.prepare(cutout_size)

    if calculate_noise and noise_model is None:
        noise_model = NoiseModel(
            image=image,
            rms=rms,
            image_conversion_factor=image_conversion_factor,
            rms_conversion_factor=rms_conversion_factor,
            uncorrelated=uncorrelated
        )
        noise_model.find_noise_square(noise_square_size, cutout_size)
        noise_model.set_noise_covariance()

    return flux(image, centers, psf_deconvolver, weight_sizes, noise_model, cutout_size=cutout_size, image_conversion_factor=image_conversion_factor, bilinear_interpolation=bilinear_interpolation, show_progress=show_progress)
