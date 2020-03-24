import numpy as np

def max_embedding_dist(ndims: int):
    # each dim must be in (-1, 1)
    # final embedding must be on unit sphere
    return 2


def perpendicular_dist(ndims: int):
    # distance between two points that represent
    # the intersection of perpendicular lines that
    # pass though the center
    return 2 ** 0.5


def receptive_field_radius(input_size: np.ndarray, output_size: np.ndarray):
    # assuming the corners of the output contain the respective
    # corners of the input in their receptive field
    return input_size / 2 - output_size / 2

def get_coordinate_scale_bounds():
    # The lower bound is set by the non_max_suppression window size
    # which defines the maximum distance between two maxima on the same process

    # The upper bound is defined by the receptive field of each output voxel
    # which defines the closest two maxima may be, without having any "knowledge"
    # of the other.

    # The farthest two neighboring maxima may be while still being on the
    # same process is twice the window size since both maxima could be in the
    # corners of their respective boxes. Given the embeddings are equal, this
    # distance should be less than alpha
    max_valid_dist = nms_window_size * 2

    # The receptive field has radius r where r is the difference between the
    # input size and the output size / 2
    min_invalid_dist = (input_size - output_size) // 2

    # max(max_valid_dist) * c < alpha
    # min(min_invalid_dist) * c > alpha
    # alpha/min(min_invalid_dist) < c < alpha / max(max_valid_dist)
    lower = alpha / min(min_invalid_dist)
    upper = alpha / max(max_valid_dist)
    return lower, upper

