from medmnistc.corruptions.registry import DATASET_RGB, CORRUPTIONS_DS

from skimage.util import montage as skimage_montage
from PIL import Image

import numpy as np
import random
import cv2
import os


class Visualizer:
    def __init__(self, 
                 medmnistc_path : str,
                 medmnist_path : str,
                 output_path : str):
        """
        Class used to plot examples of the selected corruptions.

        :param medmnistc_path: Root path of the corrupted datasets.
        :param medmnist_path: Root path of the clean datasets.
        :param output_path: Root path of the generated visualizations.
        """
        self.medmnist_path = medmnist_path
        self.medmnistc_path = medmnistc_path
        self.output_path = output_path

        self.supported_datasets = [
                'bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist',
                'octmnist', 'organamnist', 'organcmnist', 'organsmnist',
                'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist'
            ]
        
        # Annotation hyperparameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1
        self.text_offset_x = 30
        self.text_offset_y = 30
        self.rect_offset = 5

        # Create folder
        os.makedirs(self.output_path, exist_ok=True)


    def plot_extended(self, 
                      dataset_name : str = None, 
                      idx_image : int = None):
        """
        Plot an image grid (N,5) where:
            - N is the number of the designed corruptions
            - 5 represents the 5 severity levels
        
        :param dataset_name: Name of the dataset to corrupt.
                             Options: {'bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist',
                                       'octmnist', 'organamnist', 'organcmnist', 'organsmnist',
                                       'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist'}
        :param idx_image: Index of the selected image to corrupt and visualize.
                          If None, a random index will be chosen.                                       
        """
        assert dataset_name in self.supported_datasets, f"Dataset not found. Please choose one among : {self.supported_datasets}"

        output_folder = os.path.join(self.output_path,'extended')
        os.makedirs(output_folder, exist_ok=True)

        # Load clean images
        clean_test_images = np.load(os.path.join(self.medmnist_path,f'{dataset_name}_224.npz'))['test_images']
        num_images = len(clean_test_images)

        # Retrieve all corruption paths
        corruptions_path = os.listdir(os.path.join(self.medmnistc_path,dataset_name))
        
        # Setup image grid
        num_rows, num_cols = len(corruptions_path), 6
        
        # Select a random image, if not selected
        if not idx_image:
            idx_image = random.randint(0,num_images-1)
        
        # Check wether the dataset is RGB or not
        n_channels = 3 if DATASET_RGB[dataset_name] else 1

        # Init images to display
        images = []

        # Iterate over corruptions (ROWS)
        for corruption in CORRUPTIONS_DS[dataset_name].keys():

            test_images = np.load(os.path.join(self.medmnistc_path,dataset_name,f'{corruption}.npz'))['test_images']

            # Annotate the image in the first column
            corrutpion_name = corruption.split(".npz")[0].replace("_"," ")
            images.append(self._annotate_img(clean_test_images[idx_image].copy(),corrutpion_name))

            # Iterate over remaining corruption severities (COLUMNS)
            for sev_idx in range(0,5):
                idx_corr = idx_image + sev_idx * num_images
                images.append(test_images[idx_corr])

        
        # Create montage with all the selected images
        montage_arr = skimage_montage(
            images, channel_axis=3 if n_channels == 3 else None,
            grid_shape=(num_rows,num_cols),
            fill=(255,255,255)
        )

        # Store output
        filename = f'{dataset_name}_id{idx_image}.png'
        img_path = os.path.join(output_folder,filename)

        print(f'Image stored at : {img_path}')

        Image.fromarray(montage_arr).save(img_path)


    def plot_one_severity(self, 
                          dataset_name: str = None, 
                          idx_image:int = None, 
                          severity: int = 3, 
                          max_per_row: int = -1):
        """
        Plot an image along with all its corruptions in a row, with a user-specified severity. 
        
        Name of the dataset to corrupt.
                             Options: {'bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist',
                                       'octmnist', 'organamnist', 'organcmnist', 'organsmnist',
                                       'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist'}
        :param idx_image: Index of the selected image to corrupt and visualize.
                          If None, a random index will be chosen.
        :param severity: Severity of the corruptions. This will be applied to all the corruptions.
        :param max_per_row: Maximum number of corruptions to show in a row. 
                            In `num_corruptions` > `max_per_row`, multiple images are stored.
        """
        assert dataset_name in self.supported_datasets, f"Dataset not found. Please choose one among : {self.supported_datasets}"

        # Init output folder
        output_folder = os.path.join(self.output_path,'one_severity')
        os.makedirs(output_folder, exist_ok=True)

        # Select the number of channels
        n_channels = 3 if DATASET_RGB[dataset_name] else 1

        # Load the clean test set
        clean_test_images = np.load(os.path.join(self.medmnist_path,f'{dataset_name}_224.npz'))['test_images']
        num_images = len(clean_test_images)
        
        # Retrieve all designed corruptions
        corruptions_path = os.listdir(os.path.join(self.medmnistc_path,dataset_name))
        
        if not idx_image:
            idx_image = random.randint(0,num_images-1)

        # Init images to display
        images = []
        
        # Define the output grid
        num_rows, num_cols = 1, len(corruptions_path) + 1 # +1 because of the clean one        
        images.append(clean_test_images[idx_image]) # 1st image (clean one)

        # Iterate over corruptions
        for corruption in CORRUPTIONS_DS[dataset_name].keys():

            test_images = np.load(os.path.join(self.medmnistc_path,dataset_name,f'{corruption}.npz'))['test_images']
            corrutpion_name = corruption.split(".npz")[0].replace("_"," ")
            idx_corr = idx_image + (severity-1) * num_images

            # Annotate the image in the first column
            images.append(self._annotate_img(test_images[idx_corr],corrutpion_name))

        # Check if we need to decompose it into multiple images 
        if max_per_row > 0 and len(images) > max_per_row:

            clean_image = images[0] # extract clean one (it will be always shown)
            corrupted_images = images[1:] # extract corrupted images
            num_corrupted_images = len(corrupted_images) 
            num_parts = num_corrupted_images // (max_per_row-1) # define the number of plots (i.e., parts)

            for pt in range(num_parts+1):
                
                # Append clean image with the current corrupted ones
                curr_images = [clean_image] + corrupted_images[(max_per_row-1)*pt:(max_per_row-1)*pt + (max_per_row-1)]

                # Create montage with all the selected images
                montage_arr = skimage_montage(
                    curr_images, channel_axis=3 if n_channels == 3 else None,
                    grid_shape=(num_rows,max_per_row),
                    fill=(255,255,255)
                )

                # Store output
                filename = f'{dataset_name}_id{idx_image}_sev{severity}_pt{pt+1}.png'
                img_path = os.path.join(output_folder,filename)

                print(f'Image stored at : {img_path}')

                Image.fromarray(montage_arr).save(img_path)

        elif max_per_row > 0:

            # Create montage with all the selected images
            montage_arr = skimage_montage(
                images, channel_axis=3 if n_channels == 3 else None,
                grid_shape=(num_rows,max_per_row),
                fill=(255,255,255)
            )

            # Store output
            filename = f'{dataset_name}_id{idx_image}_sev{severity}.png'
            img_path = os.path.join(output_folder,filename)

            print(f'Image stored at : {img_path}')

            Image.fromarray(montage_arr).save(img_path)


        else: 

            # Create montage with all the selected images
            montage_arr = skimage_montage(
                images, channel_axis=3 if n_channels == 3 else None,
                grid_shape=(num_rows,num_cols),
                fill=(255,255,255)
            )

            # Store output
            filename = f'{dataset_name}_id{idx_image}_sev{severity}.png'
            img_path = os.path.join(output_folder,filename)

            print(f'Image stored at : {img_path}')

            Image.fromarray(montage_arr).save(img_path)


    
    def _annotate_img(self, img, text):
        """Annotate the selected image with the name of the corruption.
        Specifically, a white text over a black background is placed at:
            (text_offset_x, text_offset_y)
        
        :param img: Image to annotate (np.uint8)
        :param text: Text to add on the image.          
        """
        # Get width and height of the text box
        (text_width, text_height), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)

        # Define box coordinates
        box_coords = (
                        (self.text_offset_x - self.rect_offset, self.text_offset_y + self.rect_offset), 
                        (self.text_offset_x + text_width + self.rect_offset, self.text_offset_y - text_height - self.rect_offset)
                    )

        # Black background & white text
        cv2.rectangle(img, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
        cv2.putText(img, 
                    text, 
                    (self.text_offset_x, self.text_offset_y), 
                    self.font, 
                    self.font_scale, 
                    (255, 255, 255), 
                    self.thickness, 
                    cv2.LINE_AA)

        return np.array(img)