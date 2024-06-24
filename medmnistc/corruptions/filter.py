from .base import BaseCorruption

from scipy.ndimage import zoom as scizoom
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from io import BytesIO

import torchvision.transforms.functional as TF
import numpy as np
import cv2


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]



class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


class GaussianBlur(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            kernel_min, kernel_max = self.severity_params[0], self.severity_params[-1]
            kernel = int(np.random.uniform(low=kernel_min, high=kernel_max, size=None))
            if kernel % 2 == 0: # it must be odd
                kernel -= 1
        else:
            kernel = self.severity_params[severity]
        img = TF.gaussian_blur(img, kernel_size=kernel)
        return np.array(img).astype(np.uint8)


class MotionBlur(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            radius_min, radius_max = self.severity_params[0][0], self.severity_params[-1][0]
            sigma_min, sigma_max = self.severity_params[0][1], self.severity_params[-1][1]
            radius  = np.random.uniform(low=radius_min, high=radius_max, size=None)
            sigma = np.random.uniform(low=sigma_min, high=sigma_max, size=None)
        else:
            radius_sigma = self.severity_params[severity]
            radius, sigma = radius_sigma

        output = BytesIO()
        img.save(output, format='PNG')
        img = MotionImage(blob=output.getvalue())

        img.motion_blur(radius=radius, sigma=sigma, angle=np.random.uniform(-45, 45))
        img = cv2.imdecode(np.fromstring(img.make_blob(), np.uint8),
                        cv2.IMREAD_UNCHANGED)
        
        output.close()

        if img.shape != (224, 224):
            return np.clip(img[..., [2, 1, 0]], 0, 255).astype(np.uint8)  # BGR to RGB
        else:  # greyscale to RGB
            return np.clip(np.array([img, img, img]).transpose((1, 2, 0)), 0, 255).astype(np.uint8)
 

class DefocusBlur(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):

        if augmentation:
            radius_min, radius_max = self.severity_params[0][0], self.severity_params[-1][0]
            alias_min, alias_max = self.severity_params[0][1], self.severity_params[-1][1]
            radius  = np.random.uniform(low=radius_min, high=radius_max, size=None)
            alias = np.random.uniform(low=alias_min, high=alias_max, size=None)
        else:
            radius_alias = self.severity_params[severity]
            radius, alias = radius_alias        

        img = np.array(img) / 255.
        kernel = disk(radius=radius, alias_blur=alias)

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(img[:, :, d], -1, kernel))
        
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
        out = np.clip(channels, 0, 1) * 255

        return out.astype(np.uint8)
    

class ZoomBlur(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):

        if augmentation: # hard code
            min_factor, max_factor, = 1.0, self.severity_params[-1][-1]
            min_step = self.severity_params[0][1] - self.severity_params[0][0]
            max_step = self.severity_params[-1][1] - self.severity_params[-1][0]
            max_factor_sampled = np.random.uniform(low=min_factor, high=max_factor, size=None)
            step = np.random.uniform(low=min_step, high=max_step, size=None)
            zoom_factors = np.arange(min_factor, max_factor_sampled, step)
        else:
            zoom_factors = self.severity_params[severity]

        img = (np.array(img) / 255.).astype(np.float32)
        out = np.zeros_like(img)
        for zoom_factor in zoom_factors:
            out += clipped_zoom(img, zoom_factor)

        img = (img + out) / (len(zoom_factors) + 1)
        img = np.clip(img, 0, 1) * 255

        return img.astype(np.uint8)