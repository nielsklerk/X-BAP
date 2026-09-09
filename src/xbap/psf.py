import numpy as np
from .utils import padded_cutout_with_center, load_psf_model, polynomial_weights


class PSFDeconvolver:
    def __init__(self, psf: np.ndarray | str, eps: float = 1e-8) -> None:
        if isinstance(psf, str):
            self.varying_psf = True
            self.psf_dir = psf
            self.ft_bases_flipped = None
            self.x_zero = None
            self.x_scale = None
            self.y_zero = None
            self.y_scale = None
            self.degree = None
            self.sum_of_bases = None
        else:
            self.varying_psf = False
            self.psf = np.asarray(psf, dtype=np.float64)
        self.eps = eps
        self.psf_prefactor = None
        self.KX, self.KY = None, None

    def prepare(self, cutout_size: int) -> None:
        """
        Prepares the PSF factor in the deconvolution.

        Parameters
        ----------
        cutout_size: int
            Shape of the prefactor used for the deconvolution
        eps: float = 1e-8
            Factor for numerical stability
        """

        # Create meshgrid for the fourier modes
        ky = np.fft.fftfreq(cutout_size) * 2 * np.pi
        kx = np.fft.rfftfreq(cutout_size) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky)

        if self.varying_psf:
            # Load the PSF model
            bases, self.x_zero, self.x_scale, self.y_zero, self.y_scale, self.degree = load_psf_model(self.psf_dir)
            bases = np.asarray(bases, dtype=np.float64)

            # Calculate the sum of the bases for normalization in deconvolution
            self.sum_of_bases = np.sum(bases, axis=(-2, -1))

            # Make buffer for the fourier transform of the flipped bases
            self.ft_bases_flipped = np.empty((len(bases), *self.KX.shape), dtype=np.complex128)
            for i, base in enumerate(bases):
                # Cut/Pad the base to the cutout size
                base_padded, _ = padded_cutout_with_center(base, base.shape[0] / 2, base.shape[1] / 2, cutout_size)

                # Calculate the fourier transform of the padded base
                self.ft_bases_flipped[i] = np.fft.rfft2(base_padded[::-1, ::-1])

        else:
            # Cut/Pad the PSF to the cutout size
            psf_padded, _ = padded_cutout_with_center(
                self.psf, self.psf.shape[0] / 2, self.psf.shape[1] / 2, cutout_size)

            # Calculate the fourier transform of the padded PSF
            ft_flipped_psf = np.fft.rfft2(psf_padded[::-1, ::-1])

            # Store the prefactor used for the deconvolution
            self.psf_prefactor = np.conj(ft_flipped_psf) / (np.abs(ft_flipped_psf) ** 2 + self.eps)

    def deconvolve_weight(
            self,
            weight_fft: np.ndarray,
            x_c: float,
            y_c: float,
            shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Deconvolves the weight function using the PSF.

        Parameters
        ----------
        weight_fft: np.ndarray
            Fourier transform of the weight function.
        x_c, y_c: float, float
            Centers to interpolate the PSF model at.
        shape: tuple[int, int]
            shape of the weight function.

        Returns
        -------
        np.ndarray
            Deconvolved weight function.

        """
        
        # Change the PSF prefactor if the PSF is varying
        if self.varying_psf:
            # Calculate the weights for the PSF based on the coordinates
            weights = polynomial_weights(
                x_c, y_c,
                self.x_zero, self.x_scale,
                self.y_zero, self.y_scale,
                self.degree
            )

            # Multiply the weights with the bases and sum them
            ft_flipped_psf = np.einsum("k,kij->ij", weights, self.ft_bases_flipped)

            # Normalize the result as the PSF is not normalized
            ft_flipped_psf /= (weights @ self.sum_of_bases)

            # Calculate the prefactor used for the deconvolution
            self.psf_prefactor = np.conj(ft_flipped_psf) / (np.abs(ft_flipped_psf) ** 2 + self.eps)

        # Return the deconvolved weight function
        return np.fft.irfft2(
            self.psf_prefactor * weight_fft,
            s=shape,
        )
