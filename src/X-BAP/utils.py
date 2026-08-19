import numpy as np
from numba import njit

def padded_cutout_with_center(image: np.ndarray,
                              cx: float,
                              cy: float,
                              size: int,
                              cutout: np.ndarray | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Extract a fixed-size cutout centered on (cy, cx).
    Pads with zeros when the cutout extends beyond the image

    Parameters
    ----------
    image: np.ndarray
        Image from which the cutout is extracted
    cx, cy: float
        Coordinates of the center of the cutout in the original image
    size: int
        Size of the cutout
    cutout: np.ndarray | None = None
        Buffer to store the cutout

    Returns
    -------
    np.ndarray:
        Cutout of the image
    tuple[float, float]:
        Center coordinates in the cutout corresponding to (cx, cy)

    Raises
    ------
    ValueError
        Description of when this error is raised
    """

    h, w = image.shape
    half = size // 2

    # Integer anchor
    iy = np.int64(cy)
    ix = np.int64(cx)

    # Desired bounds in image coordinates
    y0 = iy - half
    x0 = ix - half
    y1 = y0 + size
    x1 = x0 + size

    # Overlap with image
    iy0 = max(0, y0)
    ix0 = max(0, x0)
    iy1 = min(h, y1)
    ix1 = min(w, x1)

    # Corresponding region in cutout coordinates
    cy0 = iy0 - y0
    cx0 = ix0 - x0
    cy1 = cy0 + (iy1 - iy0)
    cx1 = cx0 + (ix1 - ix0)

    # Allocate cutout
    if cutout is None:
        cutout = np.zeros((size, size))
    else:
        cutout.fill(0)

    # Insert image data
    cutout[cy0:cy1, cx0:cx1] = image[iy0:iy1, ix0:ix1]

    # Center in cutout coordinates
    cy_c = cy - y0
    cx_c = cx - x0

    return cutout, (cx_c, cy_c)


@njit(fastmath=True)
def calc_flux(weight: np.ndarray,
              image: np.ndarray) -> float:
    """
    Calculate the flux of the cutout using the weight function

    Parameters
    ----------
    weight: np.ndarray
        Weight function to be applied to the image
    image: np.ndarray
        Image for which the flux is calculated

    Returns
    -------
    float:
        Flux from the image

    Raises
    ------
    ValueError
        Weight and cutout must have the same shape
    """
    if weight.shape != image.shape:
        raise ValueError("Weight and cutout must have the same shape")

    h, w = weight.shape

    s = 0.0
    for i in range(h):
        for j in range(w):
            s += weight[i, j] * image[i, j]

    return s


@njit(fastmath=True)
def calc_flux_shift(weight: np.ndarray,
                    image: np.ndarray,
                    dx: float,
                    dy: float) -> float:
    """
    Calculate the flux of the cutout using the weight function via bilinear interpolation

    Parameters
    ----------
    weight: np.ndarray
        Weight function to be applied to the image
    image: np.ndarray
        Image for which the flux is calculated
    dx: float
        Fractional part of the x coordinate (0 <= dx <= 1)
    dy: float
        Fractional part of the y coordinate (0 <= dx <= 1)

    Returns
    -------
    float:
        Flux from the image

    Raises
    ------
    ValueError
        Weight and cutout must have the same shape
    ValueError
        Fractional coordinates must be between 0 and 1
    """
    if weight.shape != image.shape:
        raise ValueError("Weight and cutout must have the same shape")
    if not 0 <= dx <= 1 or not 0 <= dy <= 1:
        raise ValueError("Fractional coordinates must be between 0 and 1")
    h, w = weight.shape
    shifted = np.zeros_like(weight)

    a = (1 - dx) * (1 - dy)
    b = (1 - dx) * dy
    c = dx * (1 - dy)
    d = dx * dy

    s = 0.0
    for i in range(h - 1):
        for j in range(w - 1):
            shifted[i, j] = (
                a * weight[i, j]
                + b * weight[i, j + 1]
                + c * weight[i + 1, j]
                + d * weight[i + 1, j + 1]
            )
            s += image[i, j] * shifted[i, j]
    return s, shifted


@njit(fastmath=True)
def gaussian_2d(x: np.ndarray,
                y: np.ndarray,
                x0: float = 0.0,
                y0: float = 0.0,
                sigma_x: float = 1.0,
                sigma_y: float = 1.0,
                theta: float = 0.0,
                amplitude: float = 1.0) -> np.ndarray:
    """
    Make a 2d Gaussian of the form A*exp(-0.5 (x - x0)^2 / sigma_x^2 - 0.5 (y - y0)^2 / sigma_y^2)

    Parameters
    ----------
    x, y: 2D ndarray
        Coordinate grids, typically produced by np.meshgrid.

        Example:
            x = np.arange(w) - (w - 1) / 2
            y = np.arange(h) - (h - 1) / 2
            X, Y = np.meshgrid(x, y)
    x0, y0: float = 0.0
        Center of the Gaussian in the grid
    sigma_x, sigma_y: float = 1.0
        Standard deviation of the Gaussian along the principal axes
    theta: float = 0.0
        Rotation angle of the Gaussian in radians
    amplitude: float = 1.0
        Amplitude of the Gaussian

    Returns
    -------
    np.ndarray:
        2d Gaussian
    """
    dx = x - x0
    dy = y - y0

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotate coordinates into Gaussian principal-axis frame
    x_rot = cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy

    return amplitude * np.exp(
        -0.5 * (
            (x_rot / sigma_x) ** 2 +
            (y_rot / sigma_y) ** 2
        )
    )


@njit(fastmath=True)
def fourier_gaussian_2d(kx: np.ndarray,
                        ky: np.ndarray,
                        sigma_x: float = 1.0,
                        sigma_y: float = 1.0,
                        theta: float = 0.0,
                        amplitude: float = 1.0,
                        fourier_gaussian: np.ndarray | None = None) -> np.ndarray:
    """
    Calculate the Fourier Transform of a 2d Gaussian centered at (0, 0)

    Parameters
    ----------
    kx, ky: 2D ndarray
        Coordinate grids, typically produced by np.meshgrid.

        Example:
            kx = np.fft.fftfreq(w) * 2*np.pi
            ky = np.fft.rfftfreq(h) * 2*np.pi

            KX, KY = np.meshgrid(kx, ky)
    sigma_x, sigma_y: float = 1.0
        Standard deviation of the Gaussian along the principal axes
    theta: float = 0.0
        Rotation angle of the Gaussian in radians
    amplitude: float = 1.0
        Amplitude of the Gaussian
    fourier_gaussian: np.ndarray | None = None
        Buffer for the Fourier Transform

    Returns
    -------
    np.ndarray:
        Fourier Transform of the 2d Gaussian
    """

    c = np.cos(theta)
    s = np.sin(theta)

    sx2 = sigma_x * sigma_x
    sy2 = sigma_y * sigma_y

    norm = amplitude * 2.0 * np.pi * sigma_x * sigma_y

    if fourier_gaussian is None:
        fourier_gaussian = np.empty_like(kx)

    for i in range(kx.shape[0]):
        kxi = kx[i]
        kyi = ky[i]

        for j in range(kx.shape[1]):

            xr = c * kxi[j] + s * kyi[j]
            yr = -s * kxi[j] + c * kyi[j]

            r2 = sx2 * xr * xr + sy2 * yr * yr
            fourier_gaussian[i, j] = norm * np.exp(-0.5 * r2)

    return fourier_gaussian


TWOPI = 2.0*np.pi
@njit
def prepare_phase_coordinates(kx, ky):
    return -TWOPI * kx, -TWOPI * ky

@njit(fastmath=True)
def compute_phase(kx_scale, ky_scale, dx, dy):
    ax = kx_scale * dx
    ay = ky_scale * dy

    sx = np.sin(ax)
    cx = np.cos(ax)

    sy = np.sin(ay)
    cy = np.cos(ay)

    # (cx + i sx)(cy + i sy)
    return (cx * cy - sx * sy) + 1j * (sx * cy + cx * sy)


@njit
def find_best_square_coords(image, box_size):
    ny, nx = image.shape

    best_std = 1e30

    best_y = -1
    best_x = -1

    found = False

    for y in range(0, ny - box_size + 1, box_size):

        for x in range(0, nx - box_size + 1, box_size):

            total = 0.0
            nonzero = 0

            for j in range(box_size):
                for i in range(box_size):

                    value = image[y + j, x + i]

                    total += value

                    if value != 0:
                        nonzero += 1

            size = box_size * box_size

            nonzero_fraction = nonzero / size

            if nonzero_fraction < 0.5:
                continue

            mean = total / size

            var = 0.0

            for j in range(box_size):
                for i in range(box_size):

                    value = image[y + j, x + i]

                    diff = value - mean

                    var += diff * diff

            std = np.sqrt(var / size)

            if std < best_std:
                best_std = std
                best_y = y
                best_x = x
                found = True

    return best_y, best_x, found
