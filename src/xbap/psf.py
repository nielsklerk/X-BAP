import numpy as np
import pyfftw.interfaces.numpy_fft as fft
from .utils import padded_cutout_with_center

class PSFDeconvolver:
    def __init__(self, psf: np.ndarray):
        self.psf = psf
        self.psf_cache = None
        self._fft_buffer = None
        self.KX, self.KY = None, None
        self.psf_prefactor = None

    def prepare(self, cutout_size: int, eps: float = 1e-8) -> None:
        """
        Prepares the PSF factor in the deconvolution.

        Parameters
        ----------
        cutout_size: int
            Shape of the prefactor used for the deconvolution
        eps: float = 1e-8
            Factor for numerical stability
        """

        ky = fft.fftfreq(cutout_size) * 2*np.pi
        kx = fft.rfftfreq(cutout_size) * 2*np.pi

        self.KX, self.KY = np.meshgrid(kx, ky)

        psf_padded, _ = padded_cutout_with_center(
            self.psf, self.psf.shape[0]/2, self.psf.shape[1]/2, cutout_size)

        ft_psf = fft.rfft2(psf_padded[::-1, ::-1])

        self.psf_prefactor = np.conj(ft_psf) / (np.abs(ft_psf)**2 + eps)