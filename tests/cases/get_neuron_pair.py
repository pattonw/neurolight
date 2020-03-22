from pathlib import Path
import itertools

from swc_base_test import SWCBaseTest
from neurolight.gunpowder.nodes.swc_file_source import SwcFileSource
from neurolight.gunpowder.nodes.grow_labels import GrowLabels
from neurolight.gunpowder.nodes.rasterize_skeleton import RasterizeSkeleton
from neurolight.gunpowder.nodes.get_neuron_pair import GetNeuronPair
from neurolight.gunpowder.nodes.binarize_labels import BinarizeLabels
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

import numpy as np


class GetNeuronPairTest(SWCBaseTest):
    def setUp(self):
        super(GetNeuronPairTest, self).setUp()

    def _toy_swc(self, file_path: Path):
        raise NotImplementedError

    def test_get_neuron_pair(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(path, self._toy_swc_points().to_nx_graph())

        # read arrays
        swc_source = PointsKey("SWC_SOURCE")
        ensure_nonempty = PointsKey("ENSURE_NONEMPTY")
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

        data_shape = 5
        output_shape = Coordinate((data_shape, data_shape, data_shape))

        # Get points from test swc
        swc_file_source = SwcFileSource(
            path,
            [swc_source, ensure_nonempty],
            [
                PointsSpec(roi=Roi((-10, -10, -10), (31, 31, 31))),
                PointsSpec(roi=Roi((-10, -10, -10), (31, 31, 31))),
            ],
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
            + RandomLocation(ensure_nonempty=ensure_nonempty, ensure_centered=True)
        )

        pipeline = skeleton + GetNeuronPair(
            point_source=swc_source,
            nonempty_placeholder=ensure_nonempty,
            array_source=imgs,
            label_source=labels,
            points=(points_a, points_b),
            arrays=(img_a, img_b),
            labels=(labels_a, labels_b),
            seperate_by=(1, 3),
            shift_attempts=100,
            request_attempts=10,
            output_shape=output_shape,
        )

        request = BatchRequest()

        request.add(points_a, output_shape)
        request.add(points_b, output_shape)
        request.add(img_a, output_shape)
        request.add(img_b, output_shape)
        request.add(labels_a, output_shape)
        request.add(labels_b, output_shape)

        with build(pipeline):
            for i in range(10):
                batch = pipeline.request_batch(request)
                assert all(
                    [
                        x in batch
                        for x in [points_a, points_b, img_a, img_b, labels_a, labels_b]
                    ]
                )

                min_dist = 5
                for a, b in itertools.product(
                    batch[points_a].nodes,
                    batch[points_b].nodes,
                ):
                    min_dist = min(
                        min_dist,
                        np.linalg.norm(a.location - b.location),
                    )

                self.assertLessEqual(min_dist, 3)
                self.assertGreaterEqual(min_dist, 1)
