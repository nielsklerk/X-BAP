import numpy as np
from numba import njit
from astropy.io import fits


@njit(fastmath=True)
def padded_cutout_with_center(image: np.ndarray,
                              cx: float,
                              cy: float,
                              size: int,
                              cutout: np.ndarray | None = None
                              ) -> tuple[np.ndarray, tuple[float, float]]:
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
              image: np.ndarray
              ) -> float:
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
def gaussian_2d(x: np.ndarray,
                y: np.ndarray,
                x0: float = 0.0,
                y0: float = 0.0,
                sigma_x: float = 1.0,
                sigma_y: float = 1.0,
                theta: float = 0.0,
                amplitude: float = 1.0
                ) -> np.ndarray:
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
                        sigma_x: float,
                        sigma_y: float,
                        theta: float,
                        fourier_gaussian: np.ndarray
                        ) -> np.ndarray:
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
    fourier_gaussian: np.ndarray
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

    norm = 2.0 * np.pi * sigma_x * sigma_y

    for i in range(kx.shape[0]):
        kxi = kx[i]
        kyi = ky[i]

        for j in range(kx.shape[1]):
            xr = c * kxi[j] + s * kyi[j]
            yr = -s * kxi[j] + c * kyi[j]

            r2 = sx2 * xr * xr + sy2 * yr * yr
            fourier_gaussian[i, j] = norm * np.exp(-0.5 * r2)


TWOPI = 2.0 * np.pi


@njit(fastmath=True)
def prepare_phase_coordinates(kx, ky):
    return -TWOPI * kx, -TWOPI * ky


@njit(fastmath=True)
def compute_phase(kx, ky, dx, dy, out):
    nx = kx.shape[1]
    ny = ky.shape[0]

    for i in range(ny):
        ay = ky[i, 0] * dy
        sy = np.sin(ay)
        cy = np.cos(ay)

        for j in range(nx):
            ax = kx[0, j] * dx
            sx = np.sin(ax)
            cx = np.cos(ax)

            out[i, j] = (
                    (cx * cy - sx * sy)
                    + 1j * (sx * cy + cx * sy)
            )


def load_psf_model(filename):
    with fits.open(filename) as hdul:
        hdu = hdul["PSF_DATA"]
        hdr = hdu.header

        basis = hdu.data["PSF_MASK"][0]

        x_zero = hdr["POLZERO1"]
        x_scale = hdr["POLSCAL1"]

        y_zero = hdr["POLZERO2"]
        y_scale = hdr["POLSCAL2"]

        degree = hdr["POLDEG1"]

    return basis, x_zero, x_scale, y_zero, y_scale, degree


def polynomial_weights(x, y, x_zero, x_scale, y_zero, y_scale, degree):
    u = (x - x_zero) / x_scale
    v = (y - y_zero) / y_scale

    weights = []

    for total_degree in range(degree + 1):
        for i in range(total_degree + 1):
            j = total_degree - i
            weights.append(u ** i * v ** j)

    return np.asarray(weights)
