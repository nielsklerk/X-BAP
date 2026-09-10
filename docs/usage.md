# Usage

The main function in X-BAP is **xbap_flux**. This calculates the X-BAP flux and error for one or more sources. 
This is done by deconvolving a weight function in the pre-seeing image by the PSF, which allows for the same weight function to be used in bands with differenent PSFs. 

***IMPORTANT:*** The Gaussian weight function needs to be larger than the largest PSF of the bands in which X-BAP is applied.

## Basic Usage

**xbap_flux** always requires an image, psf, center and weight parameters.

```python
from xbap import xbap_flux

flux, _ = xbap_flux(image=image,
                    psf=psf,
                    centers=centers,
                    weight_sizes=5
                    )
```
The function returns the X-BAP flux for each source using a circular Gaussian with scale parameter of 5 pixels.
As there is no noise image provided, no error on the flux is calculated so None is returned instead.

## Input

### Image

**image** is a two-dimensional array that contains the observation.

### PSF

**psf** is either a two-dimensional array that contains the PSF or a path to a PSFEx .psf file.


```python
flux, _ = xbap_flux(image=image,
                    psf=psf,
                    centers=centers,
                    weight_sizes=5
                    )
```

```python
flux, _ = xbap_flux(image=image,
                    psf="path/to/.psf",
                    centers=centers,
                    weight_sizes=5
                    )
```

### Source centers

**centers** are the coordinates of the centers of sources in the image.
Multiple centers can be be provided in the following way.

```python
centers = [[100, 100],
           [200, 300],
           [300, 400]
           ]
```

The X-BAP flux is measured for each center.

### Weight parameters

**weight_sizes** determines the shape and size of the apertures used to measure the X-BAP flux.
The weight is the weight function that is applied to image in the pre-seeing image and is defined in the following way.

$$
W(x,y) = 
\exp\left[
-\frac{1}{2}
\left(
\left(\frac{x'}{\sigma_x}\right)^2 +
\left(\frac{y'}{\sigma_y}\right)^2
\right)
\right],
$$

where

$$
\Delta x = x-x_0,\;
\Delta y = y-y_0
$$

and the coordinates are rotated by an angle $\theta$:

$$
x' = \cos(\theta)\Delta x+\sin(\theta)\Delta y,\;
y' = -\sin(\theta)\Delta x+\cos(\theta)\Delta y.
$$

$x_0, y_0$ are determined by the center coordinates described above. $\sigma_x$, $\sigma_y$, and $\theta$ can be set to change the shape of the weight function.
When $\sigma_x=\sigma_y$, the aperture is circular and reduces to

$$
W(x,y)
=
\exp\left[
-\frac{1}{2}
\left(
\frac{(x-x_0)^2 + (y-y_0)^2}{\sigma^2}
\right)
\right],
$$

where $\sigma=\sigma_x=\sigma_y$. In the case that $\sigma_x\neq\sigma_y$, the weight function is elliptical.

The shape of **weight_sizes** is flexible as several shapes are accounted for.

- scalar: when a single value is provided, the same circular Gaussian weight function is applied at every center 
```python
weight_sizes = 5
```
- (3,): when a three values are provided, the same elliptical Gaussian weight function is applied at every center 
```python
weight_sizes = [3, 4, 5]
```
- (N,): when a N values are provided (where N is the number of centers), a unique circular Gaussian weight function is applied at every center 
```python
weight_sizes = [3, 4, 5, 6, 7]
```
- (N, 3): when a N values are provided (where N is the number of centers), a unique elliptical Gaussian weight function is applied at every center 
```python
weight_sizes = [[3, 4, 5],
                [6, 7, 8],
                [9, 10, 11]
                ]
```
For the elliptical Gaussian weight function the parameters are [$\sigma_x$, $\sigma_y$, $\theta$] in that order.

### Noise image
**noise** is an image that represent the noise in the image, e.g., a sourceless patch.
This is used to estimate the uncertainty on the X_BAP flux. 

```python
flux, error = xbap_flux(image=image,
                    psf=psf",
                    centers=centers,
                    weight_sizes=5,
                    noise=noise
                    )
```
When **noise** is provided, the function returns both the flux and the uncertainty. In the calculation of the uncertainty, it is assumed that the noise is sampled from a uniform RMS map which is derived from the noise image.

### RMS map

**rms** can be additionally be provided when **noise** is provided, when the noise in the image.

```python
flux, error = xbap_flux(image=image,
                    psf=psf,
                    centers=centers,
                    weight_sizes=5,
                    noise=noise,
                    rms=rms
                    )
```
In this case, the noise image is used to find the correlation between the pixel values in the image and the RMS map is used for the noise in the pixel values.

### Cutout size

In the measurement of the X-BAP flux, cutouts are made around the center of the sources. 
By default this is set to 128 pixels

```python
cutout_size = 128
```
This value can be changed to any value to change the size of the cutout. (It is recommended to choose a value that is a power of 2 as FFT is involved.)

```python
flux, error = xbap_flux(image=image,
                    psf=psf,
                    centers=centers,
                    weight_sizes=5,
                    noise=noise,
                    rms=rms,
                    cutout_size = 256
                    )
```
When a source is near the edge of the image, the image is padded with zeroes to keep the same **cutout_size**.


### Conversion factors

If the image and/or the RMS map need to be converted to different units before the measurement, conversion factors can be provided,
for example, when the image provided is in units of AB magnitude.
The image and RMS map are then multiplied by the **image_conversion_factor** and **rms_conversion_factor**, respectively.

```python
flux, error = xbap_flux(image=image,
                    psf=psf,
                    centers=centers,
                    weight_sizes=5,
                    noise=noise,
                    rms=rms,
                    image_conversion_factor=5,
                    rms_conversion_factor=5
                    )
```

### Progress bar

A progress bar can be displayed when setting **show_progress** to True

```python
flux, error = xbap_flux(image=image,
                    psf=psf,
                    centers=centers,
                    weight_sizes=5,
                    noise=noise,
                    rms=rms,
                    show_progress=True
                    )
```

### Noise correlation

In X-BAP it is not assumed that the noise is uncorrelated. However, it is possible to assume that it is uncorrelated by setting **uncorrelated** to True.
This makes the code faster as it less calculation are involved, but it can also result in the underestimation of the uncertainty.

```python
flux, error = xbap_flux(image=image,
                    psf=psf,
                    centers=centers,
                    weight_sizes=5,
                    noise=noise,
                    rms=rms,
                    uncorrelated=True
                    )
```

### Epsilon

During the deconvolution, **eps** is used to stabilize the result as it prevents division by 0.

$$
W^i_\mathrm{A} = \mathcal{F}^{-1}\left\{\frac{\mathcal{F}\{\bar{P}_i\}^*}{|\mathcal{F}\{\bar{P}_i\}|^2+\varepsilon}\cdot\mathcal{F}\{\tilde{W}_\mathrm{A}\}\right\}.
$$
