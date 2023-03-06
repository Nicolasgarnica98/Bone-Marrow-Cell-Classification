import glob
import os
import pandas as pd
import tensorflow as tf
from getdataset import get_dataset
from modelprocessing import CNN_Model, ML_Model
from sklearn.model_selection import train_test_split
from preprocessing import pre_processing ,My_Custom_Generator

def main():

    model_name = str(input('Insert a name for the model to load/train: '))
    compressed_dataset = get_dataset.download('https://1drv.ms/u/s!Aocxj1Hi_hVIldsnb2DuGP-DMhD1hw?e=DQYEld')
    get_dataset.unzip_dataset(compressed_dataset)

    #Get labels from csv
    df_lbl_csv = pd.read_csv('dataset/abbreviations.csv', sep=';')
    df_lbl = df_lbl_csv.Abbreviation.values

    #Split dataset
    if os.path.exists('./dataset/train') == False:
        df_img = glob.glob(os.path.join('dataset/bone_marrow_cell_dataset/*.jpg'))
        df_labels = get_dataset.get_labels(df_img, df_lbl)
        #Split dataset
        df_img_train, df_img_test, df_lbl_train, df_lbl_test = train_test_split(df_img,df_labels,test_size=0.2)
        df_img_train, df_img_val, df_lbl_train, df_lbl_val = train_test_split(df_img_train,df_lbl_train, test_size=0.1)
        get_dataset.divide_dataset_in_folders(df_train=df_img_train, df_test=df_img_test, df_val=df_img_val)

    #Train pipeline
    def train_pipeline(selected_model=None):
        # train_img_array = get_dataset.load_images(train_data,'train data')
        df_train = glob.glob(os.path.join('dataset/train/*.jpg'))
        df_val = glob.glob(os.path.join('dataset/val/*.jpg'))

        lbl_train = get_dataset.get_labels(df_train,df_lbl)
        lbl_val = get_dataset.get_labels(df_val,df_lbl)

        if isinstance(selected_model, CNN_Model):
            # pp_train_img_data = pre_processing.resize_images(x_size=50, y_size=50, img_array=train_img_array)
            # get_dataset.data_exploration(pp_train_img_data,lbl_train,df_lbl)
            # pp_train_img_data = pre_processing.image_normalization(pp_train_img_data)
            
            training_batch_generator = My_Custom_Generator(df_train,lbl_train,batch_size=batch_size, x_size=x_size, y_size=y_size)
            val_batch_generator = My_Custom_Generator(df_val,lbl_val,batch_size=batch_size, x_size=x_size, y_size=y_size)
            selected_model.train_model(input_shape=(x_size,y_size,3), train_labels=lbl_train, train_generator=training_batch_generator, val_generator=val_batch_generator, val_labels=lbl_val)
    

    #Decide wich model to use
    selected_model = str(input('Select model from -> CNN_Model, ML_Model: '))

    if selected_model == 'CNN_Model':
        #CNN Model, pre processing parameters:
        batch_size = 256
        x_size = 80
        y_size = 80
        actual_model = CNN_Model(model_name=model_name,epochs=35,batch_size=batch_size)

    elif selected_model =='ML_Model':
        actual_model = ML_Model

    #Decide wether to train or load the selected model
    if os.path.exists(f'./saved models/{model_name}_SavedModel.h5') == False:
        if os.path.exists('./saved models')==False:
            os.mkdir('./saved models')
            os.mkdir('./saved train-history')
            train_pipeline(df_img_train,df_lbl_train,df_img_val,df_lbl_val,actual_model)
        else:
            train_pipeline(df_img_train,df_lbl_train,df_img_val,df_lbl_val,actual_model)
    elif os.path.exists(f'./saved models/{model_name}_SavedModel.h5') == True:
        want_to_train = input(f'If you want to re-train the model "{model_name}", write True: ')
        if want_to_train == 'True':
            train_pipeline(df_img_train,df_lbl_train,df_img_val,df_lbl_val,actual_model)
    
    #Get train performance metrics
    actual_model.get_train_performance_metrics()
    #Get Model summary
    actual_model.get_model_summary()

    #Get predictions
    df_test = glob.glob(os.path.join('dataset/test/*.jpg'))
    test_img_array = get_dataset.load_images(df_test,'test images')
    pp_test_img_array = pre_processing.resize_images(x_size=x_size,y_size=y_size, img_array=test_img_array)
    pp_test_img_array = pre_processing.image_normalization(pp_test_img_array)
    lbl_test = get_dataset.get_labels(df_test,df_lbl)
    with tf.device('/CPU:0'):
        actual_model.model_prediction(pp_test_img_array,lbl_test,df_lbl)

if __name__ =='__main__':
    main()
