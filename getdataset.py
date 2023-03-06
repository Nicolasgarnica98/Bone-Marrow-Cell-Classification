import os
import wget
import base64
import shutil
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
from zipfile import ZipFile
from skimage.io import imread
import matplotlib.pyplot as plt


class get_dataset:

    def download(url):

        def create_onedrive_directdownload (onedrive_link):
            data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
            data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
            resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
            return resultUrl

        if os.path.exists('./dataset') == False:
            os.mkdir('./dataset')
            onedrive_url = url
            # Generate Direct Download URL from above Script
            direct_download_url = create_onedrive_directdownload(onedrive_url)

            print('Downloading dataset')
            r = wget.download(url=direct_download_url, out='./dataset/')
            print('\n')
            return r


    def unzip_dataset(dataset_comp):
        if dataset_comp != None:
            with ZipFile(dataset_comp,'r') as zip_object:
                zip_object.extractall(path='./dataset/')
            os.remove(dataset_comp)


    def get_labels(df_img, df_labels):
        labels = []
        for i in range(0,len(df_img)):
            for j in range(0,len(df_labels)):
                if df_img[i].find(df_labels[j])!=-1:
                    labels.append(j)

        labels = np.array(labels)

        return labels


    def load_images(df_img, dataset_name):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_array = []
        for i in tqdm(range(0,len(df_img)),desc=f'Loading {dataset_name}: '):
            img_array.append(imread(df_img[i]))
            
        return img_array
    
    def divide_dataset_in_folders(df_train, df_test, df_val):
        os.mkdir('./dataset/train')
        os.mkdir('./dataset/test')
        os.mkdir('./dataset/val')

        folder_array = ['./dataset/train', './dataset/test', './dataset/val']
        df_array = [df_train,df_test,df_val]

        for i in tqdm(range(0,len(folder_array)),'Moving files: '):
            for file in df_array[i]:
                source = file
                shutil.move(source, folder_array[i])
        if os.path.exists('./datset/bone_marrow_cell_dataset'):
            os.remove('./datset/bone_marrow_cell_dataset')

    def data_exploration(img_array, labels_txt, df_lbl_txt):
        plot_img = []
        plot_lbl = []
        for i in range(0,6):
            img_indx = np.random.randint(0,len(img_array)-1)
            img = img_array[img_indx]
            img_label_ = labels_txt[img_indx]
            img_label = df_lbl_txt[img_label_]
            plot_img.append(img)
            plot_lbl.append(img_label)
        
        fig1, ax1 = plt.subplots(2,3,figsize=(10, 10))
        fig1.suptitle('\nDataset exploration\n')
        ax1[0][0].set_title(f'Img size: {plot_img[0].shape}\n Label: {plot_lbl[0]}')
        ax1[0][0].imshow(plot_img[0], cmap='gray')
        ax1[0][0].axis('off')
        ax1[0][1].set_title(f'Img size: {plot_img[1].shape}\n Label: {plot_lbl[1]}')
        ax1[0][1].imshow(plot_img[1], cmap='gray')
        ax1[0][1].axis('off')
        ax1[0][2].set_title(f'Img size: {plot_img[2].shape}\n Label: {plot_lbl[2]}')
        ax1[0][2].imshow(plot_img[2], cmap='gray')
        ax1[0][2].axis('off')
        ax1[1][0].set_title(f'Img size: {plot_img[3].shape}\n Label: {plot_lbl[3]}')
        ax1[1][0].imshow(plot_img[3], cmap='gray')
        ax1[1][0].axis('off')
        ax1[1][1].set_title(f'Img size: {plot_img[4].shape}\n Label: {plot_lbl[4]}')
        ax1[1][1].imshow(plot_img[4], cmap='gray')
        ax1[1][1].axis('off')
        ax1[1][2].set_title(f'Img size: {plot_img[5].shape}\n Label: {plot_lbl[5]}')
        ax1[1][2].imshow(plot_img[5], cmap='gray')
        ax1[1][2].axis('off')
        fig1.tight_layout()
        plt.show()


