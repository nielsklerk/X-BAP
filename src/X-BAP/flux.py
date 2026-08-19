import numpy as np
import pyfftw.interfaces.numpy_fft as fft
import pyfftw
from tqdm import tqdm
from .utils import padded_cutout_with_center, fourier_gaussian_2d, prepare_phase_coordinates, compute_phase, calc_flux, calc_flux_shift

pyfftw.interfaces.cache.enable()


def flux(image,
         centers,
         psfdeconvolver,
         weight_sizes,
         noise_model = None,
         cutout_size: int = 128,
         image_conversion_factor: int = 1,
         bilinear_interpolation: bool = False,
         show_progress: bool = True):

    # Create FFT builder
    complex_in = pyfftw.empty_aligned(
        (cutout_size, cutout_size // 2 + 1),
        dtype="complex64",
    )
    irfft2 = pyfftw.builders.irfft2(
        complex_in
    )

    centers = np.atleast_2d(centers)
    ws = np.asarray(weight_sizes)

    Nc = centers.shape[0]

    # Normalize weight input
    # If only one weight size is given, the weight becomes a circular Gaussian
    if ws.ndim == 0:
        ws = np.array([[ws.item(), ws.item(), 0.0]])
    # If multiple weight sizes are given, the weights are expanded to match the number of centers
    elif ws.ndim == 1:
        if ws.size == 3:
            ws = ws.reshape(1, 3)
        else:
            ws = np.column_stack([ws, ws, np.zeros_like(ws)])
    elif ws.ndim == 2 and ws.shape[1] != 3:
        raise ValueError("weight_sizes must have shape (N,) or (N,3)")

    Nw = ws.shape[0]

    # Check the type of loop to be used
    if Nw == 1:
        mode = "scalar_weight"
    elif Nc == 1:
        mode = "scalar_center"
    elif Nc == Nw:
        mode = "paired"
    else:
        raise ValueError(
            "Mismatch: centers and weights must match or be scalar-expanded")

    # Initialize the FFT grid
    H = W = cutout_size
    ky = fft.fftfreq(H)[:, None]
    kx = fft.rfftfreq(W)[None, :]
    kx_scaled, ky_scaled = prepare_phase_coordinates(kx, ky)

    #
    fluxes = np.empty(max(Nc, Nw))
    variances = None if noise_model is None else np.empty(max(Nc, Nw))

    last_weight = None
    weight_fft = None
    cutout_buffer = np.empty((H, W))

    # ----------------------------
    # Main loop
    # ----------------------------
    if Nw > 1:
        sort_idx = np.argsort(ws[:, 0])
    else:
        sort_idx = np.array([0])

    for j in tqdm(range(max(Nc, Nw)), desc='Measuring Flux', disable=not show_progress):

        i_w = sort_idx[j] if j < Nw else sort_idx[0]

        # ---- select center ----
        if mode == "scalar_center":
            x_c, y_c = centers[0]
            out_idx = i_w
        elif mode == "paired":
            x_c, y_c = centers[i_w]
            out_idx = i_w
        else:  # scalar_weight
            x_c, y_c = centers[j]
            out_idx = j

        # ---- select weight ----
        if mode == "scalar_weight":
            sx, sy, th = ws[0]
        else:
            sx, sy, th = ws[i_w]

        # ---- recompute PSF-weight FFT if needed ----
        current_weight = (sx, sy, th)

        if current_weight != last_weight:
            FT = fourier_gaussian_2d(
                psfdeconvolver.KX,
                psfdeconvolver.KY,
                sx, sy, th
            )
            weight_fft = psfdeconvolver.psf_prefactor * FT
            last_weight = current_weight
            if bilinear_interpolation:
                complex_in[:] = weight_fft
                weight_rescale_unshifted = irfft2()

        # ---- extract cutout ----
        cutout, (cx_cut, cy_cut) = padded_cutout_with_center(
            image, x_c, y_c, cutout_size, cutout_buffer
        )
        cutout *= image_conversion_factor

        # ---- subpixel shift ----
        ix = int(cx_cut)
        iy = int(cy_cut)
        dx = cx_cut - ix
        dy = cy_cut - iy

        if bilinear_interpolation:
            fluxes[out_idx], weight_rescale = calc_flux_shift(
                weight_rescale_unshifted, cutout, dx, dy)

        else:
            phase = compute_phase(kx_scaled, ky_scaled, dx, dy)
            np.multiply(weight_fft, phase, out=complex_in)
            weight_rescale = irfft2()
            fluxes[out_idx] = calc_flux(weight_rescale, cutout)

        # ---- variance ----
        if noise_model is not None:
            if noise_model.rms is None:
                complex_in[:] = weight_fft
                wf = irfft2()
            else:
                wf = weight_rescale

            variances[out_idx] = noise_model.calc_error(
                wf, x_c, y_c, cutout_size
            )

    return fluxes if noise_model is None else (fluxes, variances)
