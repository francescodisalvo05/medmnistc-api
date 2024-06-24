from .base import BaseCorruption

import skimage as sk
import numpy as np


class GaussianNoise(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            c  = np.random.uniform(low=range_min, high=range_max, size=None)
        else:
            c = self.severity_params[severity]
        
        img = np.array(img) / 255.
        noisy_image = np.clip(img + np.random.normal(size=img.shape, scale=c), 0, 1)
        return (noisy_image * 255).astype(np.uint8)


class ImpulseNoise(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            c  = np.random.uniform(low=range_min, high=range_max, size=None)
            noisy_image = sk.util.random_noise(np.array(img) / 255., mode='s&p', amount=c, rng=np.random.default_rng(99999))
        else:
            c = self.severity_params[severity]
            noisy_image = sk.util.random_noise(np.array(img) / 255., mode='s&p', amount=c, rng=self.rng)
        noisy_image = np.clip(noisy_image, 0, 1)
        return (noisy_image * 255).astype(np.uint8)


class SpeckleNoise(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            c  = np.random.uniform(low=range_min, high=range_max, size=None)
        else:
            c = self.severity_params[severity]
        img = np.array(img) / 255.
        noise = np.random.normal(size=img.shape, scale=c)
        noisy_image = np.clip(img + img * noise, 0, 1)
        return (noisy_image * 255).astype(np.uint8)


class ShotNoise(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            mult  = np.random.uniform(low=range_min, high=range_max, size=None)
        else:
            mult = self.severity_params[severity]
        img = np.array(img) / 255.
        noisy_image = np.clip(np.random.poisson(img * mult) / mult, 0, 1)
        return (noisy_image * 255).astype(np.uint8)
    

