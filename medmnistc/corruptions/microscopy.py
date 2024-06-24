from .base import BaseCorruption
from PIL import Image, ImageDraw

import numpy as np
import random
import cv2
import string
import random


class StainDeposit(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):

        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            max_marks = int(np.random.uniform(low=range_min, high=range_max, size=None))
        else:
            max_marks = self.severity_params[severity]
        

        x = np.array(img) 
        img_w, img_h = x.shape[1], x.shape[0]
            
        if max_marks > 1:
            num_marks = random.randint(1,max_marks)
        else:
            num_marks = max_marks

        inks = self.inks[str(max(3,severity))] # size

        for _ in range(num_marks):

            ink = inks[random.randint(0,len(inks)-1)]
            ink_height, ink_width = ink.shape
            rand_x = random.randint(10,img_w - ink_width - 10) 
            rand_y = random.randint(10,img_h - ink_height - 10) 

            for idx in range(3): #channels
                x[rand_y: rand_y + ink_height, rand_x:rand_x + ink_width,idx] *= (1-ink) # black

        return np.clip(x, 0, 255).astype(np.uint8)
    

class Bubble(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):

        if augmentation:
            range_min_rad, range_max_rad = self.severity_params[0][0], self.severity_params[-1][0]
            range_min_bub, range_max_bub = self.severity_params[0][1], self.severity_params[-1][1]
            max_radius = int(np.random.uniform(low=range_min_rad, high=range_max_rad, size=None))
            max_bubbles = int(np.random.uniform(low=range_min_bub, high=range_max_bub, size=None))
        else:
            maxradius_bubbles = self.severity_params[severity]
            max_radius, max_bubbles = maxradius_bubbles

        height, width = img.size
        output_image = img.copy()

        # create a new image for the bubbles with the same dimensions as the original image
        # and transparent background (RGBA mode)
        bubbles_image = Image.new('RGBA', output_image.size, (255, 255, 255, 0))
        # create a drawing context for the bubble image
        draw = ImageDraw.Draw(bubbles_image)
        # define border effect of the bubble
        border = 2

        num_bubbles = random.randint(7,max_bubbles)

        # draw several bubbles
        for _ in range(num_bubbles):  
            radius = random.randint(3, max_radius)  # random radius
            x, y = random.randint(radius, width - radius), random.randint(radius, height - radius)
            alpha = 100 # transparency
            draw.ellipse((x - radius - border, y - radius - border, x + radius + border, y + radius + border), fill=(255, 255, 255, alpha + 30))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 255, 255, alpha))

        # overlay the bubbles onto the original image
        output_image.paste(bubbles_image, (0, 0), bubbles_image)
        return np.array(output_image)
    

class BlackCorner(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            multiplier = np.random.uniform(low=range_min, high=range_max, size=None)
        else:
            multiplier = self.severity_params[severity]
            
        width, height = img.size
        img = np.array(img)
        
        center = (width // 2, height // 2)
        
        # default radius
        radius = min(center[0], center[1], width - center[0], height - center[1])
        # adjust radius based on the severity / aug
        circle = np.zeros((height, width), np.uint8)
        cv2.circle(circle, center, int(radius * multiplier), (255), thickness=-1)
        mask = circle == 255
        img[~mask] = 0

        return img


class Characters(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):

        if augmentation:

            range_min_w, range_max_w = self.severity_params[0][0], self.severity_params[-1][0]
            range_min_l, range_max_l = self.severity_params[0][1], self.severity_params[-1][1]
            range_min_fs, range_max_fs = self.severity_params[0][2], self.severity_params[-1][2]
            max_words = int(np.random.uniform(low=range_min_w, high=range_max_w, size=None))
            max_letters = int(np.random.uniform(low=range_min_l, high=range_max_l, size=None))
            max_font_scale = np.random.uniform(low=range_min_fs, high=range_max_fs, size=None)

        else:

            c = self.severity_params[severity]
            max_words, max_letters, max_font_scale = c
        
        num_words = random.randint(1,max_words)

        for _ in range(num_words):

            num_letters = random.randint(3,max_letters)
            font_scale = random.randint(14,int(max_font_scale * 100)) / 100.

            letters = string.ascii_lowercase
            random_str = ''.join(random.choice(letters) for _ in range(num_letters))

            img = np.array(img)

            width, height = img.shape[1], img.shape[0]

            # randomly sample the position of the string with respect to the image
            # org = (x,y) represents the bottom left corner
            rand_x = random.randint(10,width - (8 * num_letters))
            rand_y = random.randint(10,height - 10)
            org = (rand_x,rand_y) 
        
            # black character
            color = self.random_color()
            thickness = 1
            
            img = cv2.putText(img, random_str, org, self.font,  
                        font_scale, color, thickness, cv2.LINE_AA)
        
        return img
    

    def random_color(self):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        return (red, green, blue)