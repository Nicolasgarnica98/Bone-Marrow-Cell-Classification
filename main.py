import glob
import os
import tensorflow as tf 
from getdataset import get_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # model_name = str(input('Insert a name for the model to load/train: '))
    compressed_dataset = get_dataset.download('https://1drv.ms/u/s!Aocxj1Hi_hVIldsnb2DuGP-DMhD1hw?e=DQYEld')
    get_dataset.unzip_dataset(compressed_dataset)

    df_img = glob.glob(os.path.join('dataset/bone_marrow_cell_dataset/*.jpg'))

    #Get labels from csv
    df_lbl_csv = pd.read_csv('dataset/abbreviations.csv', sep=';')
    df_lbl = df_lbl_csv.Abbreviation.values
    df_labels = get_dataset.get_labels(df_img, df_lbl)

    #Split dataset
    df_img_train, df_img_test, df_lbl_train, df_lbl_test = train_test_split(df_img,df_labels,test_size=0.2)
    
    def train_pipeline(train_data, lbl_train, model=None):
        train_img_array = get_dataset.load_images(train_data)
        get_dataset.data_exploration(train_img_array,lbl_train)

    train_pipeline(df_img_train,df_lbl_train)

    

if __name__ =='__main__':
    main()
