from gunpowder.graph_points import GraphPoint

from .swc_file_source import SwcFileSource

import numpy as np

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_transform(transform_path: Path):
    text = transform_path.open("r").read()
    lines = text.split("\n")
    constants = {}
    for line in lines:
        if len(line) > 0:
            variable, value = line.split(":")
            constants[variable] = float(value)
    spacing = (
        np.array([constants["sx"], constants["sy"], constants["sz"]])
        / 2 ** (constants["nl"] - 1)
        / 1000
    )
    origin = spacing * (
        (np.array([constants["ox"], constants["oy"], constants["oz"]]) // spacing)
        / 1000
    )
    return origin, spacing


class MouselightSwcFileSource(SwcFileSource):
    """An extension of SwcFileSource that uses a transform.txt file to transform coordinates.

    reads points into pixel space based on the transform.txt file
    """

    def __init__(self, *args, **kwargs):
        transform_file = kwargs.pop("transform_file")
        super().__init__(*args, **kwargs)
        self.transform_file = Path(transform_file)

    def _parse_swc(self, filename: Path):
        """Read one point per line. If ``ndims`` is 0, all values in one line
        are considered as the location of the point. If positive, only the
        first ``ndims`` are used. If negative, all but the last ``-ndims`` are
        used.
        """
        # initialize file specific variables
        points = {}
        header = True
        offset = np.array([0, 0, 0])

        assert self.transform_file.exists(), "Missing transform.txt file: {}".format(
            str(self.transform_file.absolute())
        )
        origin, spacing = load_transform(self.transform_file)

        # parse file
        with filename.open() as o_f:
            for line in o_f.read().splitlines():
                if header and line.startswith("#"):
                    # Search header comments for variables
                    offset = self._search_swc_header(line, "offset", offset)
                    continue
                elif line.startswith("#"):
                    # comments not in header get skipped
                    continue
                elif header:
                    # first line without a comment marks end of header
                    header = False

                row = line.strip().split()
                if len(row) != 7:
                    raise ValueError("SWC has a malformed line: {}".format(line))

                # extract data from row (point_id, type, x, y, z, radius, parent_id)
                points[int(row[0])] = GraphPoint(
                    point_id=int(row[0]),
                    parent_id=int(row[6]),
                    point_type=int(row[1]),
                    location=np.array(
                        (
                            (np.array([float(x) for x in row[2:5]]) + offset - origin)
                            / spacing
                        ).take(self.transpose)
                        * self.scale,
                        dtype=float,
                    ),
                    radius=float(row[5]),
                )
            self._add_points_to_source(points)
