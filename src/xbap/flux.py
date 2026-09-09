import numpy as np
from tqdm import tqdm
import warnings

warnings.simplefilter("always")

from .noise import NoiseModel
from .psf import PSFDeconvolver
from .utils import (
    padded_cutout_with_center,
    fourier_gaussian_2d,
    prepare_phase_coordinates,
    compute_phase,
    calc_flux,
)


def flux(image: np.ndarray,
         centers: np.ndarray,
         weight_sizes: float | np.ndarray,
         psfdeconvolver: PSFDeconvolver,
         noise_model: NoiseModel | None = None,
         cutout_size: int = 128,
         image_conversion_factor: int = 1,
         show_progress: bool = True,
         ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Calculate the X-BAP aperture flux.

    Parameters
    ----------
    image: np.ndarray
        Image in which the aperture flux will be calculated.
    centers: np.ndarray
        Centers of the Gaussian apertures.
    weight_sizes: float | np.ndarray, shape (N,) or (N, 3)
        Parameters determining the shape of the apertures, with supported shapes.

        - scalar: same scale circular Gaussian apertures is used for all centers.

        - (N,): one cicular Gaussian aperture per center.

        - (N, 3): each elliptical Gaussian aperture is defined by three parameters:
        the scale parameter along the x-axis, the scale parameter along the y-axis,
        and the rotation angle relative to the vertical axis.
    psf_deconvolver: PSFDeconvolver
        PSF deconvolution algorithm.
    noise_model: NoiseModel
        Noise model.
    cutout_size: int = 128
        Size of the cutout.
    image_conversion_factor: float = 1.0
        Conversion factor.
    show_progress: bool = False
        Whether to show the progress.

    Returns
    -------
    tuple[np.ndarray, np.ndarray|None]
        The calculated X-BAP aperture flux. When measurement errors are included,
        returns (flux, error) else (flux, None).

    """

    image = np.asarray(image, dtype=np.float64)

    centers = np.atleast_2d(centers)
    ws = np.asarray(weight_sizes)

    if centers.shape[1] != 2:
        raise ValueError("centers must have shape (N,2)")

    # Normalize weight input
    # If only one weight size is given, the weight becomes a circular Gaussian
    if ws.ndim == 0:
        ws = np.array([[ws.item(), ws.item(), 0.0]])

    elif ws.ndim == 1:
        # One elliptical Gaussian for all centers
        if ws.size == 3:
            ws = ws.reshape(1, 3)

        # One circular Gaussian for each center
        else:
            ws = np.column_stack([
                ws,
                ws,
                np.zeros_like(ws),
            ])

    elif ws.ndim == 2 and ws.shape[1] != 3:
        raise ValueError(
            "weight_sizes must have shape (N,) or (N,3)"
        )

    # Number of centers and weights
    Nc = len(centers)
    Nw = len(ws)

    # Check the type of loop to be used
    if Nw == 1:
        mode = "scalar_weight"

    elif Nc == 1:
        mode = "scalar_center"

    elif Nc == Nw:
        mode = "paired"

    else:
        raise ValueError(
            "Mismatch: centers and weight_sizes must have the same "
            "number of elements unless one of them contains only one element"
        )

    complex_in = np.empty(
        (cutout_size, cutout_size // 2 + 1),
        dtype=np.complex64,
    )

    ky = np.fft.fftfreq(cutout_size)[:, None]
    kx = np.fft.rfftfreq(cutout_size)[None, :]

    kx_scaled, ky_scaled = prepare_phase_coordinates(kx, ky)

    n_measurements = max(Nc, Nw)

    # Allocate output arrays
    fluxes = np.empty(n_measurements)
    if noise_model is None:
        variances = None
    else:
        variances = np.empty(n_measurements)

    # Reusable buffers
    last_weight = None
    weight_fft = np.empty_like(
        psfdeconvolver.KX,
        dtype=np.complex64,
    )
    cutout_buffer = np.empty(
        (cutout_size, cutout_size),
        dtype=np.float64,
    )
    phase = np.empty(
        (cutout_size, cutout_size // 2 + 1),
        dtype=np.complex64,
    )

    # Sort weights so equal weights are adjacent
    if Nw > 1:
        sort_idx = np.lexsort(
            (
                ws[:, 2],
                ws[:, 1],
                ws[:, 0],
            )
        )

    #
    negative_power_fraction = np.zeros(n_measurements)

    # Main loop
    for j in tqdm(
            range(n_measurements),
            desc="Measuring Flux",
            disable=not show_progress,
    ):

        # Select sorted index unless only one weight is given
        if Nw > 1:
            i_w = sort_idx[j]
        else:
            i_w = 0

        # Select center and output index
        if mode == "scalar_center":
            # Only one center, so the output index is the same as the weight index
            x_c, y_c = centers[0]
            out_idx = i_w
        elif mode == "paired":
            # Center and weight index are the same
            x_c, y_c = centers[i_w]
            out_idx = i_w
        else:
            # Only one weight, so the output index is the same as the center index
            x_c, y_c = centers[j]
            out_idx = j

        # Recompute the weight FFT if the weight has changed
        current_weight = ws[i_w]
        if last_weight is None or np.any(current_weight != last_weight):
            fourier_gaussian_2d(
                psfdeconvolver.KX,
                psfdeconvolver.KY,
                current_weight[0],
                current_weight[1],
                current_weight[2],
                weight_fft,
            )

            last_weight = current_weight

        # Extract cutout from image
        cutout, (cx_cut, cy_cut) = padded_cutout_with_center(
            image,
            x_c,
            y_c,
            cutout_size,
            cutout_buffer,
        )

        # Apply the conversion factor
        cutout *= image_conversion_factor

        # Find the subpixel translation in the cutout
        ix = int(cx_cut)
        iy = int(cy_cut)

        dx = cx_cut - ix
        dy = cy_cut - iy

        # If there is no subpixel translation, the FFT can be computed directly
        if dx == 0.0 and dy == 0.0:
            complex_in[:] = weight_fft

        # Otherwise, the FFT must be computed using a phase shift
        else:
            compute_phase(
                kx_scaled,
                ky_scaled,
                dx,
                dy,
                phase,
            )

            np.multiply(
                weight_fft,
                phase,
                out=complex_in,
            )

        # Find the rescaled weight based on the PSF
        weight_rescale = psfdeconvolver.deconvolve_weight(complex_in, x_c, y_c, (cutout_size, cutout_size))

        # Check if the deconvolution was succesful
        negative_power = np.sum(weight_rescale[weight_rescale < 0] ** 2)
        total = np.sum(weight_rescale ** 2)
        negative_power_fraction[j] = negative_power / total

        # Calculate the aperture flux
        fluxes[out_idx] = calc_flux(
            weight_rescale,
            cutout,
        )

        # Calculate the measurement error
        if noise_model is not None:
            variances[out_idx] = noise_model.calc_error(
                weight_rescale,
                x_c,
                y_c,
                cutout_size,
            )

    # Warn user when deconvolution was 
    bad_convolutions = np.flatnonzero(negative_power_fraction > 1e-2)
    if len(bad_convolutions) > 0:
        warnings.warn(
            f'Ringing detected for {len(bad_convolutions)} deconvolutions: increase the size of the weight to reduce error')

    # Return the fluxes and errors
    if noise_model is None:
        return fluxes, None
    return fluxes, np.sqrt(variances)
