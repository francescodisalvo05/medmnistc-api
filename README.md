# 🏥 MedMNIST-C 

We introduce MedMNIST-C [[preprint](https://arxiv.org/pdf/2406.17536)], a `benchmark dataset` based on the MedMNIST+ collection covering `12 2D datasets and 9 imaging modalities`.  We simulate task and modality-specific image corruptions of varying severity to comprehensively evaluate the robustness of established algorithms against `real-world artifacts` and `distribution shifts`. We further show that our simple-to-use artificial corruptions allow for highly performant, lightweight `data augmentation` to enhance model robustness.

<p align="center">
   <img src="assets/images/wallpaper.gif" alt="Preview of image corruptions">
</p>

## Installation and Requirements

```
pip install medmnistc
```

We do require [Wand](https://docs.wand-py.org/en/latest/guide/install.html) for image manipulation, a Python binding for [ImageMagick](https://imagemagick.org/index.php). Thus, if you are using Ubuntu:

```
sudo apt-get install libmagickwand-dev
```

otherwise, please check the [tutorial](https://docs.wand-py.org/en/0.2.4/guide/install.html).

## Main components

* `medmnistc/corruptions/registry.py`: List of all the corruptions and respective intensity hyperparameters.
* `medmnistc/dataset_manager.py`: Dataset class responsible for the creation of the corrupted datasets.
* `medmnistc/visualizer.py`: Class used to visualize and store the defined corruptions.
* `medmnistc/augmentation.py`: Augumentation class based on the defined corruptions.
* `medmnistc/dataset.py`: Dataset class used for the corrupted datasets.
* `medmnistc/eval.py`: PyTorch class used for model evaluation under corrupted datasets.
* `medmnistc/assets/baseline/*`: Normalization baselines used for model evaluation under corrupted datasets.

## Basic usage

### Create the corrupted datasets
```python
from medmnistc.dataset_manager import DatasetManager

medmnist_path = ... # PATH TO THE CLEAN IMAGES
medmnistc_path = ... # PATH TO THE CORRUPTED IMAGES

ds_manager = DatasetManager(medmnist_path = medmnist_path, output_path=output_path)
ds_manager.create_dataset(dataset_name = "breastmnist") # create a single corrupted test set
ds_manager.create_dataset(dataset_name = "all") # create all
```

### Augmentations
```python
from medmnistc.augmentation import AugMedMNISTC
from medmnistc.corruptions.registry import CORRUPTIONS_DS
import torchvision.transforms as transforms

dataset = "breastmnist" # select dataset
train_corruptions = CORRUPTIONS_DS[dataset] # load the designed corruptions for this dataset
images = ... # load images

# Augment with AugMedMNISTC
augment = AugMedMNISTC(train_corruptions)
augmented_img = augment(images[0])

# Integrate into transforms.Compose
aug_compose = transforms.Compose([
    AugMedMNISTC(train_corruptions),
    transforms.ToTensor(),
    transforms.Normalize(mean=..., std=...)
])

augmented_img = aug_compose(images[0])
```

### Notebooks

* [Create the dataset](assets/examples/create_dataset.ipynb)
* [Visualize the corruptions](assets/examples/visualize.ipynb)
* [Evaluate the corruptions](assets/examples/evaluation.ipynb)
* [Use the designed augmentations](assets/examples/augment.ipynb)

## License

The code is under [Apache-2.0 License](./LICENSE).

The MedMNIST-C dataset is licensed under Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), except DermaMNIST-C under Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).

## Citation

If you find this work useful, please consider citing us:
```
@article{disalvo2024medmnist,
  title={MedMNIST-C: Comprehensive benchmark and improved classifier robustness by simulating realistic image corruptions},
  author={Di Salvo, Francesco and Doerrich, Sebastian and Ledig, Christian},
  journal={arXiv preprint arXiv:2406.17536},
  year={2024}
}
```

`DISCLAIMER`: This repository is inspired by MedMNIST APIs and the ImageNet-C repository. Thus, please also consider citing [MedMNIST](https://www.nature.com/articles/s41597-022-01721-8), the respective source datasets (described [here](https://medmnist.com/)) and [ImageNet-C](https://arxiv.org/abs/1903.12261). 

## Release versions

* `v0.1.0`: MedMNIST-C beta release.
