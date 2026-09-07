import numpy as np
from scipy.signal import fftconvolve
from astropy.io import fits


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


# Create a 2D grid of coordinates
N = 128
x = np.arange(0, N, 1) - N / 2
y = np.arange(0, N, 1) - N / 2
X, Y = np.meshgrid(x, y)

# Galaxy image is modeled as a Gaussian
galaxy = gaussian_2d(X, Y, 0, 0, 10, 2)
galaxy /= np.sum(galaxy)
galaxy *= 100

# PSF is a Gaussian
psf = gaussian_2d(X, Y, 0, 0, 5, 5)
psf /= np.sum(psf)

# Observation is the convolution of the galaxy image and the PSF
observation = fftconvolve(galaxy, psf, mode='same')

# Add noise to the observation
noise_sigma = .1
noise = np.random.normal(0, noise_sigma, (N, N))
noise_kernel = gaussian_2d(X, Y, 0, 0, 5, 2)
noise_kernel /= np.sum(noise_kernel)
noise = fftconvolve(noise, noise_kernel, mode='same')
noisy_observation = observation + noise

# Calculate RMS map corresponding to the noise
noise_rms = noise_sigma * np.sqrt(np.sum(noise_kernel ** 2))
rms = np.ones_like(noisy_observation) * noise_rms

# Save pre-seeing image
fits.writeto(
    "data/pre_seeing.fits",
    galaxy,
    overwrite=True,
)

# Save clean observation
fits.writeto(
    "data/clean_observation.fits",
    observation,
    overwrite=True,
)

# Save noisy observation
fits.writeto(
    "data/noisy_observation.fits",
    noisy_observation,
    overwrite=True,
)

# Save noise
fits.writeto(
    "data/noise.fits",
    noise,
    overwrite=True,
)

# Save psf
fits.writeto(
    "data/psf.fits",
    psf,
    overwrite=True,
)

# Save RMS
fits.writeto(
    "data/rms.fits",
    rms,
    overwrite=True,
)
