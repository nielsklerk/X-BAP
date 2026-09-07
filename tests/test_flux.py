import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.signal import fftconvolve
from xbap import xbap_flux, NoiseModel, PSFDeconvolver


def gaussian_2d(x, y, x0=0, y0=0, sigma_x=1, sigma_y=1, theta=0, A=1):
    dx = x - x0
    dy = y - y0

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotate coordinates
    x_rot = cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy

    return A * np.exp(
        -0.5 * (
                (x_rot / sigma_x) ** 2 +
                (y_rot / sigma_y) ** 2
        )
    )


def test_flux():
    pre_seeing_image = fits.getdata("data/pre_seeing.fits")
    clean_observation = fits.getdata("data/clean_observation.fits")
    psf = fits.getdata("data/psf.fits")

    nx, ny = pre_seeing_image.shape
    x = np.arange(nx) - nx / 2
    y = np.arange(ny) - ny / 2
    X, Y = np.meshgrid(x, y)

    sigma_x, sigma_y, theta = 6, 6, np.pi / 3
    weight = gaussian_2d(X, Y, 0, 0, sigma_x, sigma_y, theta)

    # Weight function applied in the pre-seeing image
    true_aperture_flux = np.sum(pre_seeing_image * weight)

    # Weight function applied in the observation (without noise)
    measured_aperture_flux = xbap_flux(clean_observation, psf, [nx / 2, ny / 2], [sigma_x, sigma_y, theta])[0]
    assert np.isclose(true_aperture_flux, measured_aperture_flux[0])


if __name__ == "__main__":
    test_flux()
