import numpy as np
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.transform import resize

class pre_processing:
    
    def resize_images(x_size, y_size, img_array):
            rs_img_array = []
            rs_img = None
            for i in tqdm(range(0,len(img_array)),f'Resizing images to {x_size}px x {y_size}px'):
                rs_img = resize(img_array[i], (x_size,y_size))
                rs_img_array.append(rs_img)
            return rs_img_array

