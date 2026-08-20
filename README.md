# X-BAP
Cross-Band Aperture Photometry (X-BAP) is a code for measuring aperture fluxes in astronomical images. It compensates for the effects of the PSF on the observation, thereby allowing an aperture to be applied to the pre-seeing image. This opens the possibility of accurate photometric color measurements when the PSFs in two bands do not match.

## Quick Start
```python
from astropy.io import fits
from xbap import xbap_flux

image = fits.getdata(image.fits)
psf = fits.getdata(psf.fits)

flux, error = xbap_flux(image,
                        psf,
                        [128, 128],
                        [8, 4, np.pi/4]
                        )
print(flux[0])
print(error[0])
```
