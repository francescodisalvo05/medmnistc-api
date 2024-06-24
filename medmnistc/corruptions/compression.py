from .base import BaseCorruption
from PIL import Image
from io import BytesIO

import numpy as np


class Pixelate(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            resize_factor = np.random.uniform(low=range_min, high=range_max, size=None)
        else:
            resize_factor = self.severity_params[severity]
        
        
        width, height = img.size
        img = img.resize((int(width * resize_factor), int(height * resize_factor)), Image.BOX)
        img = img.resize((width, height), Image.BOX)
        return np.array(img)


class JPEGCompression(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            compression_quality = int(np.random.uniform(low=range_min, high=range_max, size=None))
        else:
            compression_quality = self.severity_params[severity]
        
        output = BytesIO()
        img.save(output, 'JPEG', quality=compression_quality)
        img = Image.open(output)
        return np.array(img)