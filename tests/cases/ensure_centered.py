from pathlib import Path

from swc_base_test import SWCBaseTest
from neurolight import SwcFileSource
from neurolight.gunpowder.nodes.rasterize_skeleton import RasterizeSkeleton
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
)
import numpy as np


class EnsureCenteredTest(SWCBaseTest):
    def setUp(self):
        super(EnsureCenteredTest, self).setUp()

    def _toy_swc(self, file_path: Path):
        raise NotImplementedError

    def test_ensure_center_non_zero(self):
        path = Path(self.path_to("test_swc_source.swc"))

        # write test swc
        self._write_swc(path, self._toy_swc_points().to_nx_graph())

        # read arrays
        swc = PointsKey("SWC")
        img = ArrayKey("IMG")
        pipeline = (
            SwcFileSource(path, [swc], [PointsSpec(roi=Roi((0, 0, 0), (11, 11, 11)))])
            + RandomLocation(ensure_nonempty=swc, ensure_centered=True)
            + RasterizeSkeleton(
                points=swc,
                array=img,
                array_spec=ArraySpec(
                    interpolatable=False,
                    dtype=np.uint32,
                    voxel_size=Coordinate((1, 1, 1)),
                ),
            )
        )

        request = BatchRequest()
        request.add(img, Coordinate((5, 5, 5)))
        request.add(swc, Coordinate((5, 5, 5)))

        with build(pipeline):
            batch = pipeline.request_batch(request)

            data = batch[img].data
            g = batch[swc]
            assert g.num_vertices() > 0

            self.assertNotEqual(data[tuple(np.array(data.shape) // 2)], 0)

