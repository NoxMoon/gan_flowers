import glob
import os
from tqdm import tqdm

image_paths = glob.glob('imagenet_flowers/*/image64/*.jpg')
image_paths += glob.glob('oxford_flowers/17flowers_image64/*.jpg')
mask_paths = glob.glob('imagenet_flowers/*/mask64/*.jpg')
mask_paths += glob.glob('oxford_flowers/17flowers_mask64/*.jpg')

for path in tqdm(image_paths):
    os.system('cp '+path+' segmentation/image64')

for path in tqdm(mask_paths):
    os.system('cp '+path+' segmentation/mask64')
