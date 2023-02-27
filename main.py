import glob
import os
from preprocessing import pre_processing
import tensorflow as tf 
from getdataset import get_dataset, My_Custom_Generator
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from modelprocessing import CNN_Model, ML_Model

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
    def train_pipeline(train_data, lbl_train, selected_model=None):
        # train_img_array = get_dataset.load_images(train_data,'train data')
        if isinstance(selected_model, CNN_Model):
            # pp_train_img_data = pre_processing.resize_images(x_size=50, y_size=50, img_array=train_img_array)
            # get_dataset.data_exploration(pp_train_img_data,lbl_train,df_lbl)
            # pp_train_img_data = pre_processing.image_normalization(pp_train_img_data)
            
            training_batch_generator = My_Custom_Generator(train_data,df_lbl_train,batch_size=batch_size)
            selected_model.train_model(input_shape=(50,50,3), train_labels=lbl_train, train_generator=training_batch_generator)
    
    #Decide wich model to use
    selected_model = str(input('Select model from -> CNN_Model, ML_Model: '))

    if selected_model == 'CNN_Model':
        batch_size = 1024
        actual_model = CNN_Model(model_name=model_name,batch_size=batch_size)
    elif selected_model =='ML_Model':
        actual_model = ML_Model

    #Decide wether to train or load the selected model
    if os.path.exists(f'./saved models/{model_name}_SavedModel.h5') == False:
        if os.path.exists('./saved models')==False:
            os.mkdir('./saved models')
            os.mkdir('./saved train-history')
            train_pipeline(df_img_train,df_lbl_train,actual_model)
        else:
            train_pipeline(df_img_train,df_lbl_train,actual_model)
    elif os.path.exists(f'./saved models/{model_name}_SavedModel.h5') == True:
        want_to_train = input(f'If you want to re-train the model "{model_name}", write True: ')
        if want_to_train == 'True':
            train_pipeline(df_img_train,df_lbl_train,actual_model)
    
    #Get train performance metrics
    actual_model.get_train_performance_metrics()


if __name__ =='__main__':
    main()
