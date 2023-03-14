import keras
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize


class pre_processing:
    
    def resize_images(x_size, y_size, img_array):
            rs_img_array = []
            rs_img = None
            for i in tqdm(range(0,len(img_array)),f'Resizing images to {x_size}px x {y_size}px: '):
                rs_img = resize(img_array[i], (x_size,y_size))
                rs_img_array.append(rs_img)
            return rs_img_array

    def image_normalization(img_array):
            norm_img_array = np.array(img_array)/255
            return norm_img_array


class My_Custom_Generator(keras.utils.Sequence) :
  
  ImageFile.LOAD_TRUNCATED_IMAGES = True
  def __init__(self, image_filenames, labels, batch_size, x_size, y_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    self.x_size = x_size
    self.y_size = y_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            resize(imread(str(file_name)), (self.x_size, self.y_size, 3))
               for file_name in batch_x])/255.0, np.array(batch_y)