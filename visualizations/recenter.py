from pathlib import Path

from .swc_base_test import SWCBaseTest
from neurolight.gunpowder.nodes.swc_file_source import SwcFileSource
from neurolight.gunpowder.nodes.grow_labels import GrowLabels
from neurolight.gunpowder.nodes.rasterize_skeleton import RasterizeSkeleton
from neurolight.gunpowder.nodes.get_neuron_pair import GetNeuronPair
from neurolight.gunpowder.nodes.binarize_labels import BinarizeLabels
from neurolight.gunpowder.nodes.recenter import Recenter
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

    def test_recenter(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(path, self._toy_swc_points())

        # read arrays
        swc_source = PointsKey("SWC_SOURCE")
        labels_source = ArrayKey("LABELS_SOURCE")
        img_source = ArrayKey("IMG_SOURCE")
        img_swc = PointsKey("IMG_SWC")
        label_swc = PointsKey("LABEL_SWC")
        imgs = ArrayKey("IMGS")
        labels = ArrayKey("LABELS")
        points_a = PointsKey("SKELETON_A")
        points_b = PointsKey("SKELETON_B")
        img_a = ArrayKey("VOLUME_A")
        img_b = ArrayKey("VOLUME_B")
        labels_a = ArrayKey("LABELS_A")
        labels_b = ArrayKey("LABELS_B")

        # Get points from test swc
        swc_file_source = SwcFileSource(
            path, [swc_source], [PointsSpec(roi=Roi((-10, -10, -10), (31, 31, 31)))]
        )
        # Create an artificial image source by rasterizing the points
        image_source = (
            SwcFileSource(
                path, [img_swc], [PointsSpec(roi=Roi((-10, -10, -10), (31, 31, 31)))]
            )
            + RasterizeSkeleton(
                points=img_swc,
                array=img_source,
                array_spec=ArraySpec(
                    interpolatable=True,
                    dtype=np.uint32,
                    voxel_size=Coordinate((1, 1, 1)),
                ),
            )
            + BinarizeLabels(labels=img_source, labels_binary=imgs)
            + GrowLabels(array=imgs, radius=0)
        )
        # Create an artificial label source by rasterizing the points
        label_source = (
            SwcFileSource(
                path, [label_swc], [PointsSpec(roi=Roi((-10, -10, -10), (31, 31, 31)))]
            )
            + RasterizeSkeleton(
                points=label_swc,
                array=labels_source,
                array_spec=ArraySpec(
                    interpolatable=True,
                    dtype=np.uint32,
                    voxel_size=Coordinate((1, 1, 1)),
                ),
            )
            + BinarizeLabels(labels=labels_source, labels_binary=labels)
            + GrowLabels(array=labels, radius=1)
        )

        skeleton = tuple()
        skeleton += (
            (swc_file_source, image_source, label_source)
            + MergeProvider()
            + RandomLocation(ensure_nonempty=swc_source, ensure_centered=True)
        )

        pipeline = (
            skeleton
            + GetNeuronPair(
                point_source=swc_source,
                array_source=imgs,
                label_source=labels,
                points=(points_a, points_b),
                arrays=(img_a, img_b),
                labels=(labels_a, labels_b),
                seperate_by=4,
                shift_attempts=100,
            )
            + Recenter(points_a, img_a, max_offset=4)
            + Recenter(points_b, img_b, max_offset=4)
        )

        request = BatchRequest()

        data_shape = 9

        request.add(points_a, Coordinate((data_shape, data_shape, data_shape)))
        request.add(points_b, Coordinate((data_shape, data_shape, data_shape)))
        request.add(img_a, Coordinate((data_shape, data_shape, data_shape)))
        request.add(img_b, Coordinate((data_shape, data_shape, data_shape)))
        request.add(labels_a, Coordinate((data_shape, data_shape, data_shape)))
        request.add(labels_b, Coordinate((data_shape, data_shape, data_shape)))

        with build(pipeline):
            batch = pipeline.request_batch(request)

        data_a = batch[img_a].data
        assert data_a[tuple(np.array(data_a.shape) // 2)] == 1
        data_a = np.pad(data_a, (1,), "constant", constant_values=(0,))
        data_b = batch[img_b].data
        assert data_b[tuple(np.array(data_b.shape) // 2)] == 1
        data_b = np.pad(data_b, (1,), "constant", constant_values=(0,))

        data_c = data_a + data_b

        data = np.array((data_a, data_b, data_c))

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

        # volshow(data)
        # input("wait")
