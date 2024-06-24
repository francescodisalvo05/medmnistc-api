import numpy as np
import cv2
import os

class BaseCorruption:
    def __init__(self, severity_params):
        self.severity_params = severity_params
        self.font = cv2.FONT_HERSHEY_DUPLEX 
        self._inks = None

    @property
    def inks(self):
        if self._inks is None:
            # Get the directory of the current file (__file__ is the path to the current file)
            current_file_dir = os.path.dirname(os.path.realpath(__file__))
            inks_path = os.path.join(current_file_dir, './../', 'assets', 'inks.npz')
            self._inks = np.load(inks_path, allow_pickle=True)
        return self._inks

    def apply(self, img):
        raise NotImplementedError("This method should be implemented by subclasses.")


