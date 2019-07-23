from gunpowder import Point
import numpy as np


class SwcPoint(Point):
    def __init__(
        self,
        point_id: int,
        point_type: int,
        location: np.ndarray,
        radius: int,
        parent_id: int,
        label_id: int = None,
    ):

        super(SwcPoint, self).__init__(location)

        self.thaw()
        self.point_id = point_id
        self.parent_id = parent_id
        self.label_id = label_id
        self.radius = radius
        self.point_type = point_type
        self.freeze()

    def copy(self):
        return SwcPoint(
            point_id=self.point_id,
            point_type=self.point_type,
            location=self.location,
            radius=self.radius,
            parent_id=self.parent_id,
            label_id=self.label_id,
        )

    def __repr__(self):
        return "({}, {})".format(self.parent_id, self.location)

