import spimagine
import daisy
import sys
import numpy as np


def to_spimagine_coords(coordinate, roi):

    # relative to ROI begin
    coordinate -= roi.get_begin()[1:]
    # relative to ROI size in [0, 1]
    coordinate /= np.array(roi.get_shape()[1:], dtype=np.float32)
    # relative to ROI size in [-1, 1]
    coordinate = coordinate * 2 - 1
    # to xyz
    return coordinate[::-1]


def inspect(raw, roi):

    print("Reading raw data...")

    raw_data = raw.to_ndarray(roi=roi, fill_value=0)
    raw_data = np.nan_to_num(raw_data)

    return spimagine.volshow(raw_data, stackUnits=raw.voxel_size[1:][::-1])


if __name__ == "__main__":

    filename = sys.argv[1]

    raw_base = daisy.open_ds(filename, "volumes/raw_base")
    labels_base = daisy.open_ds(filename, "volumes/labels_base")
    raw_add = daisy.open_ds(filename, "volumes/raw_add")
    labels_add = daisy.open_ds(filename, "volumes/labels_add")
    raw_fused = daisy.open_ds(filename, "volumes/raw_fused")
    labels_fused = daisy.open_ds(filename, "volumes/labels_fused")

    all_data = daisy.Array(
        data=np.array(
            [
                x.to_ndarray()[0, :, :, :] if len(x.data.shape) == 4 else x.to_ndarray()
                for x in [
                    raw_base,
                    labels_base,
                    raw_add,
                    labels_add,
                    raw_fused,
                    labels_fused,
                ]
            ]
        ),
        roi=daisy.Roi((0,) + raw_base.roi.get_begin(), (6,) + raw_base.roi.get_shape()),
        voxel_size=(1,) + raw_base.voxel_size,
    )

    inspect(all_data, all_data.roi)

    input()

    labels_fg = daisy.open_ds(filename, "volumes/labels_fg_bin")
    gradient = daisy.open_ds(filename, "volumes/gradient_fg")
    print(labels_fg.roi)

    all_data = daisy.Array(
        data=np.array(
            [
                x.to_ndarray()[0, :, :, :] if len(x.data.shape) == 4 else x.to_ndarray()
                for x in [labels_fg, gradient]
            ]
        ),
        roi=daisy.Roi(
            (0,) + labels_fg.roi.get_begin(), (2,) + labels_fg.roi.get_shape()
        ),
        voxel_size=(1,) + raw_base.voxel_size,
    )

    inspect(all_data, all_data.roi)

    input()
