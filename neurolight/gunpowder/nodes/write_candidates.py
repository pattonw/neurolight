from funlib import math
import gunpowder as gp
import numpy as np
import pymongo

from daisy.persistence import MongoDbGraphProvider

import logging

logger = logging.getLogger(__name__)


class WriteMaxima(gp.BatchFilter):
    def __init__(self, maxima, db_host, db_name, read_size, write_size):

        self.maxima = maxima
        self.db_host = db_host
        self.db_name = db_name
        self.client = None

        self.read_size = read_size
        self.write_size = write_size
        self.context = (read_size - write_size) / 2
        assert self.write_size + 2 * self.context == self.read_size

    def setup(self):

        # Initialize client. Doesn't the daisy mongodb graph provider handle this?
        if self.client is None:
            self.client = pymongo.MongoClient(host=self.db_host)
            self.db = self.client[self.db_name]
            create_indices = "nodes" not in self.db.list_collection_names()
            self.candidates = self.db["nodes"]
            if create_indices:
                self.candidates.create_index(
                    [(l, pymongo.ASCENDING) for l in ["t", "z", "y", "x"]],
                    name="position",
                )
                self.candidates.create_index(
                    [("id", pymongo.ASCENDING)], name="id", unique=True
                )

        self.updates(self.maxima, self.spec[self.maxima].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        assert request[self.maxima].get_shape() == self.read_size
        deps[self.maxima] = request[self.maxima].copy()
        return deps

    def process(self, batch, request):

        read_roi = batch[self.maxima].spec.roi
        voxel_size = batch[self.maxima].spec.voxel_size

        write_roi = read_roi.grow(-self.context, -self.context)

        maxima = batch[self.maxima].data

        candidates = []
        for index in np.argwhere(maxima):
            index = gp.Coordinate(index)

            position = read_roi.get_begin() + voxel_size * index
            if write_roi.contains(position):

                candidate_id = int(math.cantor_number(position // voxel_size))

                candidates.append(
                    {
                        "id": candidate_id,
                        "z": position[0],
                        "y": position[1],
                        "x": position[2],
                    }
                )

        if len(candidates) > 0:
            self.candidates.insert_many(candidates)


class WriteGraph(gp.BatchFilter):
    def __init__(
        self,
        mst,
        db_host,
        db_name,
        distance_attr,
        read_size,
        write_size,
        voxel_size,
        directed,
        mode="r+",
    ):

        self.mst = mst
        self.db_host = db_host
        self.db_name = db_name
        self.client = None

        self.read_size = read_size
        self.write_size = write_size
        self.context = (read_size - write_size) / 2
        assert self.write_size + 2 * self.context == self.read_size

        self.directed = directed
        self.mode = "r"

    def setup(self):

        # Initialize client. Doesn't the daisy mongodb graph provider handle this?
        if self.client is None:
            self.client = pymongo.MongoClient(host=self.db_host)
            self.db = self.client[self.db_name]
            create_indices = "nodes" not in self.db.list_collection_names()
            self.candidates = self.db["nodes"]
            if create_indices:
                self.candidates.create_index(
                    [(l, pymongo.ASCENDING) for l in ["t", "z", "y", "x"]],
                    name="position",
                )
                self.candidates.create_index(
                    [("id", pymongo.ASCENDING)], name="id", unique=True
                )

        self.updates(self.mst, self.spec[self.mst].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        assert request[self.mst].get_shape() == self.read_size
        deps[self.mst] = request[self.mst].copy()
        return deps

    def process(self, batch, request):

        read_roi = batch[self.mst].spec.roi
        voxel_size = batch[self.mst].spec.voxel_size

        write_roi = read_roi.grow(-self.context, -self.context)

        self.client.read_

        if len(candidates) > 0:
            self.candidates.insert_many(candidates)
