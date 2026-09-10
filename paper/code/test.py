import numpy as np
import matplotlib.pyplot as plt
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

# Galaxy image is modeled as a Gaussian
galaxy = gaussian_2d(X, Y, 0, 0, 20, 5, np.pi / 3)
galaxy /= np.sum(galaxy)
galaxy *= 1000

# PSF is a Gaussian
psf = gaussian_2d(X, Y, 0, 0, 1, 1)
psf /= np.sum(psf)

# Observation is the convolution of the galaxy image and the PSF
observation = fftconvolve(galaxy, psf, mode='same')

# Weight parameters
sigma_x, sigma_y, theta = 3, 3, np.pi / 3

# Noise correlation kernel
noise_kernel = gaussian_2d(X, Y, 0, 0, 5, 1)
noise_kernel /= np.sum(noise_kernel)

# RMS map
rms_map = gaussian_2d(X, Y, 0, 0, 10, 5, np.pi / 3) * .5  # Gaussian RMS map
rms_map += np.ones_like(observation) * 0.1

# Monte Carlo simulation
N_simulation = 100000
flux = np.zeros(N_simulation)
error = np.zeros((N_simulation, 2))
for i in range(N_simulation):
    # White noise
    noise = np.random.normal(0, 1, (N, N))

    # Correlate noise with the noise correlation kernel
    noise = fftconvolve(noise, noise_kernel, mode='same')

    # Make the noise have the same RMS as the RMS map
    noise /= np.sqrt(np.sum(noise_kernel ** 2))
    noise *= rms_map

    # Add noise to the observation
    noisy_observation = observation + noise

    # Calculate the flux and error using just the noise information
    results = xbap_flux(noisy_observation, psf, [N / 2, N / 2], [sigma_x, sigma_y, theta], noise)
    flux[i], error[i, 0] = results[0][0], results[1][0]

    # Calculate the flux and error using the RMS map and noise information
    results = xbap_flux(noisy_observation, psf, [N / 2, N / 2], [sigma_x, sigma_y, theta], noise, rms=rms_map)
    error[i, 1] = results[1][0]

# Plot example of the noisy observation
plt.imshow(noisy_observation, cmap='gray', origin='lower', extent=[-N / 2, N / 2, -N / 2, N / 2], aspect='auto')
plt.axis('off')
plt.show()

bins = 100
plt.figure(figsize=(10, 5))
run = ['Non RMS', 'RMS']
for i in range(2):
    plt.hist(error[:, i], density=True, label=f'Measured Error {run[i]}', bins=bins, alpha=0.5)
plt.axvline(x=np.std(flux), color='r', label='True Error')
plt.xlabel('Error [a.u.]')
plt.ylabel('Frequency [a.u.]')
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()

weight = gaussian_2d(X, Y, 0, 0, sigma_x, sigma_y, theta)
true_aperture_flux = np.sum(galaxy * weight)

plt.figure(figsize=(10, 5))
plt.hist(flux, label='Measured Flux', bins=bins)
plt.axvline(x=true_aperture_flux, color='r', label='True Flux')
plt.xlabel('Flux [a.u.]')
plt.ylabel('Frequency [a.u.]')
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.show()