from skimage import transform, io
from skimage.exposure import adjust_gamma
import numpy as np

class random_crop():
    
    def __init__(self, image=None):
        if image is not None:
            self.get_param(image)
        else:
            self.i0 = 0
            self.i1 = 0
            self.j0 = 0
            self.j1 = 0
            
    def get_param(self, image):
        H = image.shape[0]
        W = image.shape[1]
        L = min(H,W)
        size = np.random.randint(int(0.8*L), L)
        self.i0 = np.random.randint(0, H-size)
        self.j0 = np.random.randint(0, W-size)
        self.i1 = self.i0 + size
        self.j1 = self.j0 + size
        
    def __call__(self, image):
        assert(self.i1 < image.shape[0])
        assert(self.j1 < image.shape[1])
        return image[self.i0:self.i1, self.j0:self.j1]
    
    
def process_image_n_mask(image, mask, size=(64,64), flip=False, gamma=1):
    #random crop
    crop = random_crop(image)
    image_processed = crop(image)
    mask_processed = crop(mask)
    #resize
    image_processed = transform.resize(image_processed, size)
    mask_processed = transform.resize(mask_processed, size)
    #flip
    if flip:
        image_processed = np.fliplr(image_processed)
        mask_processed = np.fliplr(mask_processed)
    #addjust gamma
    if gamma != 1:
        image_processed = adjust_gamma(image_processed, gamma)
    return image_processed, mask_processed

def process_image(image, size=(64,64), flip=False, gamma=1):
    #random crop
    crop = random_crop(image)
    image_processed = crop(image)
    #resize
    image_processed = transform.resize(image_processed, size)
    #flip
    if flip:
        image_processed = np.fliplr(image_processed)
    #addjust gamma
    if gamma != 1:
        image_processed = adjust_gamma(image_processed, gamma)
    return image_processed