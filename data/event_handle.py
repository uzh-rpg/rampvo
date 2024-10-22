import h5py
import hdf5plugin
import numpy as np
from pathlib import Path
from .events import Events


class H5EventHandle:
    def __init__(self, handle, height=None, width=None):
        assert "events" in handle.keys()
        assert "x" in handle["events"].keys()
        assert "y" in handle["events"].keys()
        assert "t" in handle["events"].keys()
        assert "p" in handle["events"].keys()
        assert "height" in handle["events"].keys()
        assert "width" in handle["events"].keys()

        self.height = height
        self.width = width
        self.height = handle["events"]["height"][()]
        self.width = handle["events"]["width"][()]

        self.handle = handle
        
    @property
    def t(self):
        return self.handle['events/t']

    @property
    def x(self):
        return self.handle['events/x']

    @property
    def y(self):
        return self.handle['events/y']

    @property
    def p(self):
        return self.handle['events/p']

    @classmethod
    def from_path(cls, path: Path, height=None, width=None):
        handle = h5py.File(str(path))
        return cls(handle, height=height, width=width)

    def get_between_idx(self, i0, i1):
        return Events(x=self.handle["events"]["x"][i0:i1],
                      y=self.handle["events"]["y"][i0:i1],
                      t=self.handle["events"]["t"][i0:i1],
                      p=self.handle["events"]["p"][i0:i1],
                      height=self.height,
                      width=self.width)

    def __len__(self):
        return len(self.handle["events"]["t"])
