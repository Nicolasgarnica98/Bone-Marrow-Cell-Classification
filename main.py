import glob
import os
import tensorflow as tf 
from Get_dataset import get_dataset


def main():
    # model_name = str(input('Insert a name for the model to load/train: '))
    compressed_dataset = get_dataset.download('https://1drv.ms/u/s!Aocxj1Hi_hVIldsnb2DuGP-DMhD1hw?e=DQYEld')
    get_dataset.unzip_dataset(compressed_dataset)

if __name__ =='__main__':
    main()
