from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import os


class CorruptedMedMNIST(Dataset):
    def __init__(self, 
                 dataset_name : str, 
                 corruption : str,
                 norm_mean : list = [0.5], 
                 norm_std : list = [0.5],
                 root : str = None,
                 as_rgb : bool = True,
                 mmap_mode : str = None):
        """
        Dataset class of CorruptedMedMNIST

        :param dataset_name: Name of the reference medmnist dataset.
        :param corruption: Name of the desired corruption.
        :param norm_mean: Normalization mean.
        :param norm_std: Normalization standard deviation.
        :param root: Root path of the generated corrupted data.
        :param as_rgb: Flag for RGB of Greyscale data.
        :param mmap_mode: Memory mapping of the file: {None, ‘r+’, ‘r’, ‘w+’, ‘c’}.
                          If not None, then memory-map the file, using the given mode 
                          (see numpy.memmap for a detailed description of the modes). 
                          Memory mapping is especially useful for accessing small 
                          fragments of large files without reading the entire file into memory.
                          src: https://numpy.org/doc/stable/reference/generated/numpy.load.html
        
        This dataset class was greatly inspired from the MedMNIST APIs:
            https://github.com/MedMNIST/MedMNIST
        """
        
        super(CorruptedMedMNIST, self).__init__()

        self.dataset_name = dataset_name
        self.corruption = corruption
        self.root = root
        self.as_rgb = as_rgb
        
        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )
        
        if not os.path.exists(os.path.join(self.root, self.dataset_name, f"{corruption}.npz")):
            print(os.path.join(self.root, self.dataset_name, f"{corruption}.npz"))
            raise RuntimeError(
                "Dataset not found."
            )

        npz_file = np.load(
            os.path.join(self.root, self.dataset_name, f"{corruption}.npz"),
            mmap_mode=mmap_mode,
        )
        
        self.imgs = npz_file["test_images"]
        self.labels = npz_file["test_labels"]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std) 
        ])
    
    
    def __len__(self):
        return self.imgs.shape[0]
        
    
    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target