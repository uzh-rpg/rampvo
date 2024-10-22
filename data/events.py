from dataclasses import dataclass
import numpy as np

from pathlib import Path

TYPES = dict(x=np.uint16, y=np.uint16, t=np.int64, p=np.int8)


@dataclass(frozen=False)
class Events:
    def __init__(self, x, y, t, p, width, height):
        self.x = x
        self.y = y
        self.t = t
        self.p = p
        self.width = width
        self.height = height

        for k, t in TYPES.items():
            if not k in ['x', 'y']:
                assert getattr(self, k).dtype == t, f"Field {k} does not have type {t}, but {getattr(self, k).dtype}."

        assert self.x.shape == self.y.shape == self.p.shape == self.t.shape
        assert self.x.ndim == 1

        if self.x.size > 0:
            assert np.max(self.p) <= 1
            self.p[self.p == 0] = -1
            assert np.max(self.x) <= self.width - 1, np.max(self.x)
            assert np.max(self.y) <= self.height - 1, np.max(self.y)
            assert np.min(self.x) >= 0
            assert np.min(self.y) >= 0

    def __len__(self):
        return len(self.x)

    def to_dict(self, format="xytp"):
        return {k: getattr(self, k) for k in format}

    def to_array(self, format="xytp"):
        return np.stack([getattr(self, k) for k in format], axis=-1)

    def __getitem__(self, item):
        return Events(x=self.x[item].copy(),
                      y=self.y[item].copy(),
                      t=self.t[item].copy(),
                      p=self.p[item].copy(),
                      width=self.width,
                      height=self.height,
                      divider=self.divider)
