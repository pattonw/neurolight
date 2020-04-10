import gunpowder as gp
import h5py
import numpy as np


class SnapshotSource(gp.BatchProvider):
    def __init__(self, snapshot_file, datasets):
        self.snapshot_file = snapshot_file
        self.datasets = datasets

    def setup(self):
        data = h5py.File(self.snapshot_file)
        print(f"{self.snapshot_file} provides:")
        for key, path in self.datasets.items():
            x = data[path]
            spec = self.spec_from_dataset(x)
            print(f"{key}: roi: {spec.roi}")
            self.provides(key, spec)

    def provide(self, request):
        outputs = gp.Batch()
        data = h5py.File(self.snapshot_file)
        for key, path in self.datasets.items():
            x = data[path]
            result = self.result_from_dataset(x)
            outputs[key] = result
        outputs = outputs.crop(request)
        return outputs

    def spec_from_dataset(self, x):
        offset = x.attrs["offset"]
        voxel_size = x.attrs["resolution"]
        spatial_dims = len(voxel_size)
        shape = x.shape[-spatial_dims:] * voxel_size
        return gp.ArraySpec(roi=gp.Roi(offset, shape), voxel_size=voxel_size)

    def result_from_dataset(self, x):
        spec = self.spec_from_dataset(x)
        return gp.Array(np.array(x), spec)
