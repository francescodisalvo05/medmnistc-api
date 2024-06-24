import numpy as np


class AugMedMNISTC(object):
    def __init__(self, 
                 train_corruptions : dict = {},
                 verbose: bool = False):
        """
        Augmentation class based on the designed image corruptions. 
        For each call, it will randomly choose *one* corruption (i.e., augmentation) using a
        uniformly sampled intensity hyperparameter in the range [min_intensity,max_intensity].
        Notably, among the possible augmentations, we do include `identity` (i.e., no aug).

        :param train_corruptions: Dictionary containing the corruptions to use during training.
        :param verbose: If True, print the name of the selected corruption.
        """
        assert len(train_corruptions) > 0, f"You need to define some corruptions firsts."
        
        self.verbose = verbose
        self.train_corruptions = train_corruptions
        self.train_corruptions_keys = list(self.train_corruptions.keys()) + ['identity']


    def __call__(self, img):
        corr = np.random.choice(self.train_corruptions_keys)

        if self.verbose:
            print(corr)

        if corr == 'identity':
            return img
        
        return self.train_corruptions[corr].apply(img, augmentation=True)


