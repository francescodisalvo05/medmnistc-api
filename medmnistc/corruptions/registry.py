from .noise import GaussianNoise, ImpulseNoise, SpeckleNoise, ShotNoise
from .compression import JPEGCompression, Pixelate
from .filter import MotionBlur, DefocusBlur, ZoomBlur, GaussianBlur
from .enhance import Brightness, Contrast, Saturate, GammaCorrection
from .microscopy import Bubble, StainDeposit, BlackCorner, Characters

import numpy as np


CORRUPTIONS_DS = {

    'pathmnist' : {
        'pixelate': Pixelate(severity_params=[0.8, 0.6, 0.40, 0.30, 0.25]),
        'jpeg_compression' : JPEGCompression(severity_params=[50, 30, 15, 10, 7]),
        'defocus_blur' : DefocusBlur(severity_params=[(3, 0.1), (4, 0.1), (5, 0.2), (6,0.2), (7, 0.3)]),
        'motion_blur' : MotionBlur(severity_params=[(5,5), (10, 5), (15, 5), (15, 8), (15, 12)]),
        'brightness_up' : Brightness(severity_params=[1.1, 1.15, 1.2, 1.22, 1.25]),
        'brightness_down' : Brightness(severity_params=[0.85, 0.80, 0.75, 0.72, 0.70]),
        'contrast_up' : Contrast(severity_params=[1.1, 1.2, 1.3, 1.4, 1.6]),
        'contrast_down' : Contrast(severity_params=[0.8, 0.7, 0.6, 0.55, 0.5]),
        'saturate' : Saturate(severity_params=[0.05, 0.10, 0.15, 0.20, 0.25]),
        'stain_deposit' : StainDeposit(severity_params=[1,2,3,4,5]),
        'bubble' : Bubble(severity_params=[(7,15),(10,15),(12,15),(15,20),(17,25)])
    }, 


    'bloodmnist' : {
        'pixelate': Pixelate(severity_params=[0.6, 0.5, 0.40, 0.30, 0.25]),
        'jpeg_compression' : JPEGCompression(severity_params=[50, 30, 15, 10, 7]),
        'defocus_blur' : DefocusBlur(severity_params=[(2, 0.01), (3, 0.1), (4,0.1), (5,0.1), (6, 0.1)]),
        'motion_blur' : MotionBlur(severity_params=[(3,3), (5,5), (10, 5), (10,7), (10, 9)]),
        'brightness_up' : Brightness(severity_params=[1.1, 1.2, 1.3, 1.35, 1.4]),
        'brightness_down' : Brightness(severity_params=[0.9, 0.8, 0.7, 0.6, 0.5]),
        'contrast_up' : Contrast(severity_params=[1.1, 1.15, 1.2, 1.25, 1.3]), 
        'contrast_down' : Contrast(severity_params=[0.9, 0.8, 0.7, 0.6, 0.5]),
        'saturate' : Saturate(severity_params=[0.05, 0.10, 0.15, 0.17, 0.20]),
        'stain_deposit' : StainDeposit(severity_params=[1,2,3,3,3]),
        'bubble' : Bubble(severity_params=[(5,10),(7,10),(10,10),(12,12),(15,12)])
    }, 


    'dermamnist' : {
        'pixelate': Pixelate(severity_params=[0.7, 0.5, 0.40, 0.30, 0.25]),
        'jpeg_compression' : JPEGCompression(severity_params=[30, 20, 15, 10, 7]),
        'gaussian_noise' : GaussianNoise(severity_params=[0.04, .08, .12, 0.18, 0.26]),
        'speckle_noise' : SpeckleNoise(severity_params=[0.05, 0.15, 0.2, 0.35, 0.45]),
        'impulse_noise' : ImpulseNoise(severity_params=[0.01, 0.03, 0.06, 0.09, 0.17]),
        'shot_noise' : ShotNoise(severity_params=[60, 25, 18, 10, 5]),
        'defocus_blur' : DefocusBlur(severity_params=[(4, 0.1), (5, 0.2), (6, 0.3), (7, 0.4), (8,0.5)]),
        'motion_blur' : MotionBlur(severity_params=[(10, 5), (15, 5), (15, 8), (15, 12), (20, 15)]),
        'zoom_blur' : ZoomBlur(severity_params=[
                        np.arange(1, 1.11, 0.01),
                        np.arange(1, 1.16, 0.01),
                        np.arange(1, 1.21, 0.02),
                        np.arange(1, 1.26, 0.02),
                        np.arange(1, 1.31, 0.03)
                      ]),
        'brightness_up' : Brightness(severity_params=[1.1, 1.2, 1.3, 1.4, 1.5]),
        'brightness_down' : Brightness(severity_params=[0.9, 0.8, 0.7, 0.6, 0.5]),
        'contrast_up' : Contrast(severity_params=[1.1, 1.2, 1.3, 1.4, 1.6]),
        'contrast_down' : Contrast(severity_params=[0.8, 0.7, 0.6, 0.5, 0.4]),
        'black_corner' : BlackCorner(severity_params=[1.10, 1.05, 1.00, 0.90, 0.95]),
        'characters' : Characters(severity_params=[(1,6,0.14),(2,7,0.15),(3,8,0.16),(4,9,0.17),(6,10,0.18)])
    },


    'retinamnist' : {
        'pixelate': Pixelate(severity_params=[0.8, 0.60, 0.50, 0.40, 0.35]),
        'jpeg_compression' : JPEGCompression(severity_params=[30, 25, 20, 10, 5]),
        'gaussian_noise' : GaussianNoise(severity_params=[0.04, 0.08, 0.12, 0.16, 0.20]),
        'speckle_noise' : SpeckleNoise(severity_params=[0.10, 0.15, 0.20, 0.25, 0.30]),
        'defocus_blur' : DefocusBlur(severity_params=[(4, 0.1), (5, 0.2), (6, 0.3), (7, 0.4), (8,0.5), (9,0.6)]),
        'motion_blur' : MotionBlur(severity_params=[(8, 5), (15, 5), (15, 8), (15, 12), (20, 15)]),
        'brightness_down' : Brightness(severity_params=[0.9, 0.8, 0.7, 0.6, 0.5]),
        'contrast_down' : Contrast(severity_params=[0.9, 0.8, 0.7, 0.6, 0.4]),
    },
    

    'tissuemnist' : {
        'pixelate': Pixelate(severity_params=[0.40, 0.30, 0.20, 0.15, 0.10]),
        'jpeg_compression' : JPEGCompression(severity_params=[25, 20, 15, 10, 7]),
        'impulse_noise' : ImpulseNoise(severity_params=[0.01, 0.015, 0.02, 0.025, 0.03]),
        'gaussian_blur' : GaussianBlur(severity_params=[13, 15, 17, 21, 25]),
        'brightness_up' : Brightness(severity_params=[1.3, 1.4, 1.5, 1.6, 1.7]),
        'brightness_down' : Brightness(severity_params=[0.8, 0.7, 0.6, 0.5, 0.4]),
        'contrast_up' : Contrast(severity_params=[1.1, 1.2, 1.3, 1.4, 1.6]),
        'contrast_down' : Contrast(severity_params=[0.9, 0.8, 0.7, 0.6, 0.4]),
    },
    

    'octmnist' : {
        'pixelate': Pixelate(severity_params=[0.30, 0.25, 0.20, 0.15, 0.10]), #
        'jpeg_compression' : JPEGCompression(severity_params=[30, 15, 10, 7, 5]),
        'speckle_noise' : SpeckleNoise(severity_params=[0.15, 0.30, 0.40, 0.50, 0.60]),
        'defocus_blur' : DefocusBlur(severity_params=[(0.5, 0.6), (1, 0.5), (1.5, 0.1), (2.0,0.5), (2.5,0.1)]),
        'motion_blur' : MotionBlur(severity_params=[(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]),
        'contrast_down' : Contrast(severity_params=[0.6, 0.4, 0.3, 0.2, 0.15])
    },


    'breastmnist' : {
        'pixelate': Pixelate(severity_params=[0.30, 0.25, 0.20, 0.15, 0.10]),
        'jpeg_compression' : JPEGCompression(severity_params=[50, 30, 15, 10, 7]),
        'speckle_noise' : SpeckleNoise(severity_params=[0.10, 0.15, 0.20, 0.25, 0.30]),
        'motion_blur' : MotionBlur(severity_params=[(5,5), (9, 7), (9,10), (13, 10), (17, 12)]),
        'brightness_up' : Brightness(severity_params=[1.4, 1.5, 1.6, 1.8, 2.0]),
        'brightness_down' : Brightness(severity_params=[0.55, 0.5, 0.45, 0.4, 0.3]),
        'contrast_down' : Contrast(severity_params=[0.9, 0.8, 0.7, 0.6, 0.4]),
    },


    'chestmnist' : {
        'pixelate': Pixelate(severity_params=[0.30, 0.25, 0.20, 0.15, 0.10]),
        'jpeg_compression' : JPEGCompression(severity_params=[50, 30, 15, 10, 7]),
        'gaussian_noise' : GaussianNoise(severity_params=[0.04, .08, .12, 0.18, 0.26]),
        'speckle_noise' : SpeckleNoise(severity_params=[0.05, 0.15, 0.2, 0.35, 0.45]),
        'impulse_noise' : ImpulseNoise(severity_params=[0.01, 0.03, 0.06, 0.09, 0.17]),
        'shot_noise' : ShotNoise(severity_params=[60, 25, 18, 10, 5]),
        'gaussian_blur' : GaussianBlur(severity_params=[3, 5, 7, 9, 11, 13]),
        'brightness_up' : Brightness(severity_params=[1.1, 1.2, 1.3, 1.4, 1.5]),
        'brightness_down' : Brightness(severity_params=[0.9, 0.8, 0.7, 0.6, 0.5]),
        'contrast_up' : Contrast(severity_params=[1.1, 1.2, 1.3, 1.4, 1.6]),
        'contrast_down' : Contrast(severity_params=[0.9, 0.8, 0.7, 0.6, 0.4]),
        'gamma_corr_up' : GammaCorrection(severity_params=[1.1, 1.2, 1.3, 1.4, 1.6]),
        'gamma_corr_down' : GammaCorrection(severity_params=[0.9, 0.8, 0.7, 0.6, 0.4]),
    },


    'pneumoniamnist' : {
        'pixelate': Pixelate(severity_params=[0.8, 0.7, 0.6, 0.5, 0.40]),
        'jpeg_compression' : JPEGCompression(severity_params=[50, 30, 15, 10, 7]),
        'gaussian_noise' : GaussianNoise(severity_params=[0.04, 0.05, 0.06, 0.07, 0.08]),
        'speckle_noise' : SpeckleNoise(severity_params=[0.05, 0.07, 0.10, 0.15, 0.20]),
        'impulse_noise' : ImpulseNoise(severity_params=[0.005, 0.01, 0.013, 0.017, 0.02]),
        'shot_noise' : ShotNoise(severity_params=[300, 200, 150, 100, 80]),
        'gaussian_blur' : GaussianBlur(severity_params=[3, 5, 7, 9, 11, 13]),
        'brightness_up' : Brightness(severity_params=[1.1, 1.2, 1.3, 1.4, 1.5]),
        'brightness_down' : Brightness(severity_params=[0.9, 0.8, 0.7, 0.6, 0.5]),
        'contrast_up' : Contrast(severity_params=[1.1, 1.2, 1.3, 1.4, 1.6]),
        'contrast_down' : Contrast(severity_params=[0.9, 0.8, 0.7, 0.6, 0.4]),
        'gamma_corr_up' : GammaCorrection(severity_params=[1.1, 1.2, 1.3, 1.4, 1.6]),
        'gamma_corr_down' : GammaCorrection(severity_params=[0.9, 0.8, 0.7, 0.6, 0.4]),
    }, 


    'organamnist' : {
        'pixelate': Pixelate(severity_params=[0.7, 0.6, 0.5, 0.40, 0.35]),
        'jpeg_compression' : JPEGCompression(severity_params=[50, 30, 15, 10, 7]),
        'gaussian_noise' : GaussianNoise(severity_params=[0.04, 0.08, 0.12, 0.16, 0.20]),
        'speckle_noise' : SpeckleNoise(severity_params=[0.05, 0.10, 0.20, 0.30, 0.40]),
        'impulse_noise' : ImpulseNoise(severity_params=[0.01, 0.02, 0.03, 0.05, 0.08]),
        'shot_noise' : ShotNoise(severity_params=[200, 100, 50, 25, 15]),
        'gaussian_blur' : GaussianBlur(severity_params=[11, 13, 15, 17, 21]), 
        'brightness_up' : Brightness(severity_params=[1.2, 1.3, 1.4, 1.5, 1.6]),
        'brightness_down' : Brightness(severity_params=[0.8, 0.75, 0.7, 0.65, 0.60]),
        'contrast_up' : Contrast(severity_params=[1.3, 1.4, 1.6, 1.7, 1.8]), 
        'contrast_down' : Contrast(severity_params=[0.8, 0.7, 0.6, 0.55, 0.5]), 
        'gamma_corr_up' : GammaCorrection(severity_params=[1.3, 1.4, 1.6, 1.8, 2.0]),
        'gamma_corr_down' : GammaCorrection(severity_params=[0.9, 0.8, 0.7, 0.6, 0.4]),
    }, 


    'organcmnist' : {
        'pixelate': Pixelate(severity_params=[0.7, 0.6, 0.5, 0.40, 0.35]),
        'jpeg_compression' : JPEGCompression(severity_params=[50, 30, 15, 10, 7]),
        'gaussian_noise' : GaussianNoise(severity_params=[0.04, 0.08, 0.12, 0.16, 0.20]),
        'speckle_noise' : SpeckleNoise(severity_params=[0.05, 0.10, 0.20, 0.30, 0.40]),
        'impulse_noise' : ImpulseNoise(severity_params=[0.01, 0.02, 0.03, 0.05, 0.08]),
        'shot_noise' : ShotNoise(severity_params=[200, 100, 50, 25, 15]),
        'gaussian_blur' : GaussianBlur(severity_params=[11, 13, 15, 17, 21]), 
        'brightness_up' : Brightness(severity_params=[1.2, 1.3, 1.4, 1.5, 1.6]),
        'brightness_down' : Brightness(severity_params=[0.8, 0.75, 0.7, 0.65, 0.60]),
        'contrast_up' : Contrast(severity_params=[1.3, 1.4, 1.6, 1.7, 1.8]), 
        'contrast_down' : Contrast(severity_params=[0.8, 0.7, 0.6, 0.55, 0.5]), 
        'gamma_corr_up' : GammaCorrection(severity_params=[1.3, 1.4, 1.6, 1.8, 2.0]),
        'gamma_corr_down' : GammaCorrection(severity_params=[0.9, 0.8, 0.7, 0.6, 0.4]),
    }, 


    'organsmnist' : {
        'pixelate': Pixelate(severity_params=[0.7, 0.6, 0.5, 0.40, 0.35]),
        'jpeg_compression' : JPEGCompression(severity_params=[50, 30, 15, 10, 7]),
        'gaussian_noise' : GaussianNoise(severity_params=[0.04, 0.08, 0.12, 0.16, 0.20]),
        'speckle_noise' : SpeckleNoise(severity_params=[0.05, 0.10, 0.20, 0.30, 0.40]),
        'impulse_noise' : ImpulseNoise(severity_params=[0.01, 0.02, 0.03, 0.05, 0.08]),
        'shot_noise' : ShotNoise(severity_params=[200, 100, 50, 25, 15]),
        'gaussian_blur' : GaussianBlur(severity_params=[11, 13, 15, 17, 21]), 
        'brightness_up' : Brightness(severity_params=[1.2, 1.3, 1.4, 1.5, 1.6]),
        'brightness_down' : Brightness(severity_params=[0.8, 0.75, 0.7, 0.65, 0.60]),
        'contrast_up' : Contrast(severity_params=[1.3, 1.4, 1.6, 1.7, 1.8]), 
        'contrast_down' : Contrast(severity_params=[0.8, 0.7, 0.6, 0.55, 0.5]), 
        'gamma_corr_up' : GammaCorrection(severity_params=[1.3, 1.4, 1.6, 1.8, 2.0]),
        'gamma_corr_down' : GammaCorrection(severity_params=[0.9, 0.8, 0.7, 0.6, 0.4]),
    }, 
}


CORRUPTIONS_DS_FOLDS = {
    'digital' : ['pixelate','jpeg_compression'],
    'noise' : ['gaussian_noise', 'speckle_noise', 'impulse_noise', 'shot_noise'],
    'blur': ['defocus_blur','motion_blur','zoom_blur','gaussian_blur'],
    'color' : ['brightness_up', 'brightness_down', 'contrast_up', 'contrast_down','saturate'],
    'task-specific' : ['stain_deposit', 'bubble','black_corner', 'characters','gamma_corr_up','gamma_corr_down']
}


DATASET_RGB = {
    'bloodmnist': True,
    'breastmnist': False,
    'chestmnist': False,
    'dermamnist': True,
    'octmnist': False,
    'organamnist': False,
    'organcmnist': False,
    'organsmnist': False,
    'pathmnist': True,
    'pneumoniamnist': False,
    'retinamnist': True,
    'tissuemnist': False
}