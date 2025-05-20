import numpy as np

class Transforms:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def transform(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x
