import glob
import os
from preprocessing import pre_processing
import tensorflow as tf 
from getdataset import get_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from modelprocessing import CNN_Model

def main():

    model_name = str(input('Insert a name for the model to load/train: '))
    compressed_dataset = get_dataset.download('https://1drv.ms/u/s!Aocxj1Hi_hVIldsnb2DuGP-DMhD1hw?e=DQYEld')
    get_dataset.unzip_dataset(compressed_dataset)

    df_img = glob.glob(os.path.join('dataset/bone_marrow_cell_dataset/*.jpg'))

    #Get labels from csv
    df_lbl_csv = pd.read_csv('dataset/abbreviations.csv', sep=';')
    df_lbl = df_lbl_csv.Abbreviation.values
    df_labels = get_dataset.get_labels(df_img, df_lbl)

    #Split dataset
    df_img_train, df_img_test, df_lbl_train, df_lbl_test = train_test_split(df_img,df_labels,test_size=0.2)
    
    #Train pipeline
    def train_pipeline(train_data, lbl_train, selec_model=None):
        actual_model = None
        train_img_array = get_dataset.load_images(train_data,'train data')
        if selec_model == 'CNN_model':
            actual_model = CNN_Model(model_name=model_name)
            pp_train_img_data = pre_processing.resize_images(x_size=50, y_size=50, img_array=train_img_array)
            get_dataset.data_exploration(pp_train_img_data,lbl_train,df_lbl)
            actual_model.train_model(pp_train_img_data, lbl_train)
        
    selec_model = input('Select model from -> CNN_Model, ML_Model: ')
    

    if os.path.exists(f'./saved models/{model_name}_SavedModel.h5') == False:
        if os.path.exists('./saved models')==False:
            os.mkdir('./saved models')
            os.mkdir('./saved train-history')
            train_pipeline(df_img_train,df_lbl_train,selec_model)
        else:
            train_pipeline(df_img_train,df_lbl_train,selec_model)
    elif os.path.exists(f'./saved models/{model_name}_SavedModel.h5') == True:
        want_to_train = input(f'If you want to re-train the model "{model_name}", write True: ')
        if want_to_train == 'True':
            train_pipeline(df_img_train,df_lbl_train,selec_model)
   

if __name__ =='__main__':
    main()
