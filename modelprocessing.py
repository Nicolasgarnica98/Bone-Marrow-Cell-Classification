import os
import numpy as np
import seaborn as sn
import tensorflow as tf
from keras import regularizers
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten


class CNN_Model:
    def __init__(self, model_name, epochs, batch_size=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        pass


    def train_model(self, input_shape, train_labels, train_generator, val_generator, val_labels):

        num_classes = len(np.unique(train_labels))
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        i = Input(shape=input_shape)

        x = Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same', kernel_regularizer=regularizers.l2(l=0.001))(i)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        
        x = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(l=0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

        # x = Conv2D(filters=128, kernel_size=(3,3), activation='relu',padding='same',kernel_regularizer=regularizers.l2(l=0.0001))(i)
        # x = BatchNormalization()(x)
        # x = MaxPooling2D(pool_size=(2,2))(x)

        x = Flatten()(x)
        x = Dense(units=1024, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(units=256,activation='relu')(x)
        x = Dropout(0.25)(x)
        # x = Dense(units=64,activation='relu')(x)
        x = Dense(num_classes, activation='softmax')(x)

        model = Model(i,x)
        model.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        result = model.fit(train_generator, epochs=self.epochs, steps_per_epoch=int(len(train_labels)//self.batch_size), validation_data=val_generator, validation_steps=int(len(val_labels)//self.batch_size))
        model.save(f'./saved models/{self.model_name}_SavedModel.h5')
        np.save(f'./saved train-history/{self.model_name}_SavedTrainHistory.npy',result.history)


    def get_train_performance_metrics(self):
        
        ModelHistory =np.load(f'./saved train-history/{self.model_name}_SavedTrainHistory.npy',allow_pickle='TRUE').item()
        fig1, ax1=plt.subplots(1,2)
        fig1.suptitle(f'{self.model_name} evaluation')
        ax1[0].set_title('Accuracy per epoch')
        ax1[0].plot(ModelHistory['accuracy'],label='Accuracy')
        ax1[0].plot(ModelHistory['val_accuracy'],label='Val Accuracy')
        ax1[0].set_xlabel('Epoch')
        ax1[0].set_ylabel('Accuracy')
        ax1[0].legend()
        ax1[0].grid(True)
        ax1[1].plot(ModelHistory['loss'],label='Loss')
        ax1[1].plot(ModelHistory['val_loss'],label='Val Loss')
        ax1[1].set_title('Loss per epoch')
        ax1[1].set_xlabel('Epoch')
        ax1[1].set_ylabel('Loss')
        ax1[1].legend()
        ax1[1].grid(True)
        fig1.tight_layout()
        plt.show()


    def model_prediction(self, img_array, lbl, lbl_txt):
        model = tf.keras.models.load_model(f'./saved models/{self.model_name}_SavedModel.h5')
        prob_predictions = model.predict(img_array)
        predictions = prob_predictions.argmax(axis=1)
        predictions.astype(int)
        cm = confusion_matrix(lbl,predictions)
        plt.title('Confusion matrix')
        sn.set(font_scale=1.4)
        x_axis_labels = lbl_txt
        y_axis_labels = lbl_txt
        conf = sn.heatmap(cm, annot=True, annot_kws={'size':8}, cmap='Blues',xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        conf.set(xlabel='Predicted class', ylabel='True class')
        conf.tick_params(left=True, bottom=True)
        metrics = classification_report(lbl,predictions)
        print(metrics)
        plt.tight_layout()
        plt.show()

    def get_model_summary(self):
        model = tf.keras.models.load_model(f'./saved models/{self.model_name}_SavedModel.h5')
        print(model.summary())


class ML_Model:
    def __init__(self):
        pass