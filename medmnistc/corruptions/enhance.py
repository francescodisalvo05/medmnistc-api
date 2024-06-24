from .base import BaseCorruption
from PIL import ImageEnhance
import skimage as sk
import numpy as np

import torchvision.transforms.functional as TF


class Brightness(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            brightness_factor  = np.random.uniform(low=range_min, high=range_max, size=None)
        else:
            brightness_factor = self.severity_params[severity]
        enhancer = ImageEnhance.Brightness(img)
        brightened_img = enhancer.enhance(brightness_factor)
        return np.array(brightened_img).astype(np.uint8)


class Contrast(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            contrast_factor  = np.random.uniform(low=range_min, high=range_max, size=None)
        else:
            contrast_factor = self.severity_params[severity]
        enhancer = ImageEnhance.Contrast(img)
        contrasted_img = enhancer.enhance(contrast_factor)
        return np.array(contrasted_img).astype(np.uint8)


class GammaCorrection(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            correction_factor  = np.random.uniform(low=range_min, high=range_max, size=None)
        else:
            correction_factor = self.severity_params[severity]
        
        img = TF.adjust_gamma(img, correction_factor, gain=1)
        return np.array(img).astype(np.uint8)


class Saturate(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            saturation_factor  = np.random.uniform(low=range_min, high=range_max, size=None)
        else:
            saturation_factor = self.severity_params[severity]
        
        img = np.array(img) / 255.
        img = sk.color.rgb2hsv(img)
        img[:, :, 1] = np.clip(img[:, :, 1] + saturation_factor, 0, 1)
        img = sk.color.hsv2rgb(img)
        img = np.clip(img, 0, 1) * 255
        
        return img.astype(np.uint8)

