from pathlib import Path

from .swc_base_test import SWCBaseTest
from neurolight.gunpowder.swc_file_source import SwcFileSource
from neurolight.gunpowder.grow_labels import GrowLabels
from neurolight.gunpowder.rasterize_skeleton import RasterizeSkeleton
from neurolight.gunpowder.get_neuron_pair import GetNeuronPair
from neurolight.gunpowder.binarize_labels import BinarizeLabels
from gunpowder import (
    ArrayKey,
    ArraySpec,
    PointsKey,
    PointsSpec,
    BatchRequest,
    Roi,
    build,
    Coordinate,
    RandomLocation,
    MergeProvider,
)

from spimagine import volshow
import numpy as np


class EnsureCentered(SWCBaseTest):
    def setUp(self):
        super(EnsureCentered, self).setUp()

    def _toy_swc(self, file_path: Path):
        raise NotImplementedError

    def test_get_neuron_pair(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(path, self._toy_swc_points())

        # read arrays
        swc_source = PointsKey("SWC")
        img_source = ArrayKey("IMG")
        imgswc = PointsKey("IMGSWC")
        label = ArrayKey("LABEL")
        points_a = PointsKey("SKELETON_A")
        points_b = PointsKey("SKELETON_B")
        img_a = ArrayKey("VOLUME_A")
        img_b = ArrayKey("VOLUME_B")

        fused_points = PointsKey("OUT_POINTS")
        fused_image = ArrayKey("OUT_ARRAY")

        # Get points from test swc
        swc_file_source = SwcFileSource(
            path, [swc_source], [PointsSpec(roi=Roi((-10, -10, -10), (31, 31, 31)))]
        )
        # Create an artificial image source by rasterizing the points
        image_source = (
            SwcFileSource(
                path, [imgswc], [PointsSpec(roi=Roi((-10, -10, -10), (31, 31, 31)))]
            )
            + RasterizeSkeleton(
                points=imgswc,
                array=img_source,
                array_spec=ArraySpec(
                    interpolatable=True,
                    dtype=np.uint32,
                    voxel_size=Coordinate((1, 1, 1)),
                ),
            )
            + BinarizeLabels(labels=img_source, labels_binary=label)
            + GrowLabels(array=label, radius=0)
        )

        skeleton = tuple()
        skeleton += (
            (swc_file_source, image_source)
            + MergeProvider()
            + RandomLocation(ensure_nonempty=swc_source, ensure_centered=True)
        )

        pipeline = skeleton + GetNeuronPair(
            point_source=swc_source,
            array_source=label,
            points=(points_a, points_b),
            arrays=(img_a, img_b),
            seperate_by=4,
            shift_attempts=10,
            request_attempts=10,
        )

        request = BatchRequest()

        data_shape = 9

        request.add(points_a, Coordinate((data_shape, data_shape, data_shape)))
        request.add(points_b, Coordinate((data_shape, data_shape, data_shape)))
        request.add(img_a, Coordinate((data_shape, data_shape, data_shape)))
        request.add(img_b, Coordinate((data_shape, data_shape, data_shape)))

        with build(pipeline):
            batch = pipeline.request_batch(request)

        data_a = batch[img_a].data
        data_a = np.pad(data_a, (1,), "constant", constant_values=(0,))
        data_b = batch[img_b].data
        data_b = np.pad(data_b, (1,), "constant", constant_values=(0,))

        data_c = data_a + data_b

        data = np.array((data_a, data_b, data_c))

        print(batch[points_a].data)
        print(batch[points_b].data)

        for _, point in batch[points_a].data.items():
            assert (
                data[(0,) + tuple(int(x) + 1 for x in point.location)] == 1
            ), "data at {} is not 1, its {}".format(
                point.location, data[(0,) + tuple(int(x) for x in point.location)]
            )

        for _, point in batch[points_b].data.items():
            assert (
                data[(1,) + tuple(int(x) + 1 for x in point.location)] == 1
            ), "data at {} is not 1".format(point.location)

        volshow(data)
        input("wait")
