import glob
import os
import tensorflow as tf 
from Get_dataset import get_dataset
from PreProcessing import pre_processing
from model_processing import CNN_Model


def main():
    model_name = str(input('Insert a name for the model to load/train: '))
    compressed_dataset = get_dataset.download('https://1drv.ms/u/s!Aocxj1Hi_hVIldsmWIMj9AcU2MH7hw?e=Z4FxiE')
    get_dataset.unzip_dataset(compressed_dataset)

if __name__ =='__main__':
    main()
