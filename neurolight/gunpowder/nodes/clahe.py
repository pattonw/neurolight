import numpy as np

from gunpowder.batch_request import BatchRequest
from gunpowder.batch import Batch
from gunpowder.array import Array
from gunpowder.coordinate import Coordinate

from gunpowder import BatchFilter

import itertools


class scipyCLAHE(BatchFilter):
    """Node utilizing scikit-images CLAHE implementation:
    https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist

    Args:

        arrays (List, :class:`ArrayKey`):

            The arrays to modify.

        kernel_size (int or array_like :int:):

            See scikit documentation

        clip_limit (float):

            See scikit documentation

        nbins (int):

            See scikit documentation
    """

    def __init__(
        self,
        arrays,
        kernel_size,
        output_arrays=None,
        clip_limit=0.01,
        nbins=256,
        context=None,
        normalize=True,
    ):
        self.arrays = arrays
        self.output_arrays = output_arrays
        if self.output_arrays is not None:
            assert len(output_arrays) == len(arrays)
        else:
            self.output_arrays = self.arrays
        self.kernel_size = np.array(kernel_size)
        self.clip_limit = clip_limit
        self.nbins = nbins
        self.context = context
        self.normalize = normalize

    def setup(self):
        self.enable_autoskip()
        for in_key, out_key in zip(self.arrays, self.output_arrays):
            if out_key == in_key:
                self.updates(out_key, self.spec[in_key])
            else:
                self.provides(out_key, self.spec[in_key])

    def prepare(self, request):
        deps = BatchRequest()
        for in_key, out_key in zip(self.arrays, self.output_arrays):
            spec = request[out_key].copy()
            if self.context is not None:
                spec.roi = spec.roi.grow(self.context, self.context)
            deps[in_key] = spec
        return deps

    def process(self, batch, request):
        output = Batch()

        for in_key, out_key in zip(self.arrays, self.output_arrays):
            array = batch[in_key]
            data = array.data
            d_min = data.min()
            d_max = data.max()
            assert (
                d_min >= 0 and d_max <= 1
            ), f"Clahe expects data in range (0,1), got ({d_min}, {d_max})"
            if np.isclose(d_max, d_min):
                output[out_key] = Array(data, array.spec)
                continue
            if self.normalize:
                data = (data - d_min) / (d_max - d_min)
            shape = data.shape
            data_dims = len(shape)
            kernel_dims = len(self.kernel_size)
            extra_dims = data_dims - kernel_dims
            voxel_size = array.spec.voxel_size

            for index in itertools.product(*[range(s) for s in shape[:extra_dims]]):
                data[index] = clahe(
                    data[index],
                    kernel_size=Coordinate(self.kernel_size / voxel_size),
                    clip_limit=self.clip_limit,
                    nbins=self.nbins,
                )
            assert (
                data.min() >= 0 and data.max() <= 1
            ), f"Clahe should output data in range (0,1), got ({data.min()}, {data.max()})"
            output[out_key] = Array(data, array.spec).crop(request[out_key].roi)
        return output


"""
Adapted code from "Contrast Limited Adaptive Histogram Equalization" by Karel
Zuiderveld <karel@cv.ruu.nl>, Graphics Gems IV, Academic Press, 1994.

http://tog.acm.org/resources/GraphicsGems/

The Graphics Gems code is copyright-protected.  In other words, you cannot
claim the text of the code as your own and resell it. Using the code is
permitted in any program, product, or library, non-commercial or commercial.
Giving credit is not required, though is a nice gesture.  The code comes as-is,
and if there are any flaws or problems with any Gems code, nobody involved with
Gems - authors, editors, publishers, or webmasters - are to be held
responsible.  Basically, don't be a jerk, and remember that anything free
comes with no guarantee.
"""
import numpy as np
from skimage.exposure import rescale_intensity


NR_OF_GRAY = 2 ** 14  # number of grayscale levels to use in CLAHE algorithm


def clahe(image, kernel_size, clip_limit, nbins):
    """Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : (N1,...,NN) ndarray
        Input image.
    kernel_size: int or N-tuple of int
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float
        Normalized clipping limit (higher values give more contrast).
    nbins : int
        Number of gray bins for histogram ("data range").

    Returns
    -------
    out : (N1,...,NN) ndarray
        Equalized image.

    The number of "effective" graylevels in the output image is set by `nbins`;
    selecting a small value (eg. 128) speeds up processing and still produce
    an output image of good quality. The output image will have the same
    minimum and maximum value as the input image. A clip limit smaller than 1
    results in standard (non-contrast limited) AHE.
    """

    ########################
    # added by me, image needs to be uint16
    # (scaling here is data independent,
    # assumes float input with values in [0, 1])
    image = np.round(
        rescale_intensity(
            image, in_range=(image.min(), image.max()), out_range=(0, NR_OF_GRAY - 1)
        )
    ).astype(np.uint16)
    ########################

    ndim = image.ndim
    dtype = image.dtype

    # pad the image such that the shape in each dimension
    # - is a multiple of the kernel_size and
    # - is preceded by half a kernel size
    pad_start_per_dim = [k // 2 for k in kernel_size]

    try:
        pad_end_per_dim = [
            (k - s % k) % k + int(np.ceil(k / 2.0))
            for k, s in zip(kernel_size, image.shape)
        ]
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"Trying to use kernel size: {kernel_size} on image with shape: {image.shape}"
        ) from e

    image = np.pad(
        image,
        [[p_i, p_f] for p_i, p_f in zip(pad_start_per_dim, pad_end_per_dim)],
        mode="reflect",
    )

    # determine gray value bins
    bin_size = 1 + NR_OF_GRAY // nbins
    lut = np.arange(NR_OF_GRAY)
    lut //= bin_size

    image = lut[image]

    # calculate graylevel mappings for each contextual region
    # rearrange image into flattened contextual regions
    ns_hist = [int(s / k) - 1 for s, k in zip(image.shape, kernel_size)]
    hist_blocks_shape = np.array([ns_hist, kernel_size]).T.flatten()
    hist_blocks_axis_order = np.array(
        [np.arange(0, ndim * 2, 2), np.arange(1, ndim * 2, 2)]
    ).flatten()
    hist_slices = [slice(k // 2, k // 2 + n * k) for k, n in zip(kernel_size, ns_hist)]
    hist_blocks = image[tuple(hist_slices)].reshape(hist_blocks_shape)
    hist_blocks = np.transpose(hist_blocks, axes=hist_blocks_axis_order)
    hist_block_assembled_shape = hist_blocks.shape
    hist_blocks = hist_blocks.reshape((np.product(ns_hist), -1))

    # Calculate actual clip limit
    if clip_limit > 0.0:
        clim = int(np.clip(clip_limit * np.product(kernel_size), 1, None))
    else:
        clim = NR_OF_GRAY  # Large value, do not clip (AHE)

    hist = np.apply_along_axis(np.bincount, -1, hist_blocks, minlength=nbins)
    hist = np.apply_along_axis(clip_histogram, -1, hist, clip_limit=clim)
    hist = map_histogram(hist, 0, NR_OF_GRAY - 1, np.product(kernel_size))
    hist = hist.reshape(hist_block_assembled_shape[:ndim] + (-1,))

    # duplicate leading mappings in each dim
    map_array = np.pad(hist, [[1, 1] for _ in range(ndim)] + [[0, 0]], mode="edge")

    # Perform multilinear interpolation of graylevel mappings
    # using the convention described here:
    # https://en.wikipedia.org/w/index.php?title=Adaptive_histogram_
    # equalization&oldid=936814673#Efficient_computation_by_interpolation

    # rearrange image into blocks for vectorized processing
    ns_proc = [int(s / k) for s, k in zip(image.shape, kernel_size)]
    blocks_shape = np.array([ns_proc, kernel_size]).T.flatten()
    blocks_axis_order = np.array(
        [np.arange(0, ndim * 2, 2), np.arange(1, ndim * 2, 2)]
    ).flatten()
    blocks = image.reshape(blocks_shape)
    blocks = np.transpose(blocks, axes=blocks_axis_order)
    blocks_flattened_shape = blocks.shape
    blocks = np.reshape(blocks, (np.product(ns_proc), np.product(blocks.shape[ndim:])))

    # calculate interpolation coefficients
    coeffs = np.meshgrid(
        *tuple([np.arange(k) / k for k in kernel_size[::-1]]), indexing="ij"
    )
    coeffs = [np.transpose(c).flatten() for c in coeffs]
    inv_coeffs = [1 - c for dim, c in enumerate(coeffs)]

    # sum over contributions of neighboring contextual
    # regions in each direction
    result = np.zeros(blocks.shape, dtype=np.float32)
    for iedge, edge in enumerate(np.ndindex(*([2] * ndim))):

        edge_maps = map_array[tuple([slice(e, e + n) for e, n in zip(edge, ns_proc)])]
        edge_maps = edge_maps.reshape((np.product(ns_proc), -1))

        # apply map
        edge_mapped = np.take_along_axis(edge_maps, blocks, axis=-1)

        # interpolate
        edge_coeffs = np.product(
            [[inv_coeffs, coeffs][e][d] for d, e in enumerate(edge[::-1])], 0
        )

        result += (edge_mapped * edge_coeffs).astype(result.dtype)

    result = result.astype(dtype)

    # rebuild result image from blocks
    result = result.reshape(blocks_flattened_shape)
    blocks_axis_rebuild_order = np.array(
        [np.arange(0, ndim), np.arange(ndim, ndim * 2)]
    ).T.flatten()
    result = np.transpose(result, axes=blocks_axis_rebuild_order)
    result = result.reshape(image.shape)

    # undo padding
    unpad_slices = tuple(
        [
            slice(p_i, s - p_f)
            for p_i, p_f, s in zip(pad_start_per_dim, pad_end_per_dim, image.shape)
        ]
    )
    result = result[unpad_slices]

    ########################
    # added by me, image needs to be converted back to [0, 1]
    result = result / (NR_OF_GRAY - 1)
    # (this undoes the scaling at the beginning of this function)
    ########################

    return result


def clip_histogram(hist, clip_limit):
    """Perform clipping of the histogram and redistribution of bins.

    The histogram is clipped and the number of excess pixels is counted.
    Afterwards the excess pixels are equally redistributed across the
    whole histogram (providing the bin count is smaller than the cliplimit).

    Parameters
    ----------
    hist : ndarray
        Histogram array.
    clip_limit : int
        Maximum allowed bin count.

    Returns
    -------
    hist : ndarray
        Clipped histogram.
    """
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit
    hist[excess_mask] = clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = n_excess // hist.size  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = np.logical_and(hist >= upper, hist < clip_limit)
    mid = hist[mid_mask]
    n_excess += mid.sum() - mid.size * clip_limit
    hist[mid_mask] = clip_limit

    while n_excess > 0:  # Redistribute remaining excess
        prev_n_excess = n_excess
        for index in range(hist.size):
            under_mask = hist < clip_limit
            step_size = max(1, np.count_nonzero(under_mask) // n_excess)
            under_mask = under_mask[index::step_size]
            hist[index::step_size][under_mask] += 1
            n_excess -= np.count_nonzero(under_mask)
            if n_excess <= 0:
                break
        if prev_n_excess == n_excess:
            break

    return hist


def map_histogram(hist, min_val, max_val, n_pixels):
    """Calculate the equalized lookup table (mapping).

    It does so by cumulating the input histogram.
    Histogram bins are assumed to be represented by the last array dimension.

    Parameters
    ----------
    hist : ndarray
        Clipped histogram.
    min_val : int
        Minimum value for mapping.
    max_val : int
        Maximum value for mapping.
    n_pixels : int
        Number of pixels in the region.

    Returns
    -------
    out : ndarray
       Mapped intensity LUT.
    """
    out = np.cumsum(hist, axis=-1).astype(float)
    out *= (max_val - min_val) / n_pixels
    out += min_val
    np.clip(out, a_min=None, a_max=max_val, out=out)

    return out.astype(int)

