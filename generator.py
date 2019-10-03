import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2
from mask2rle import rle2maskResize
from PIL import Image

def create_test_gen(test_df):
        return ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
            test_df,
            directory='../input/test_images',
            x_col='ImageId',
            class_mode=None,
            target_size=(256, 256),
            batch_size=64,
            shuffle=False)

class DataGenerator(keras.utils.Sequence):
    def __init__(self,df,batch_size=16,subset='train',shuffle=False,preprocess=None,info={}):
        super().__init__()
        self.batch_size = batch_size
        self.df = df
        self.shuffle = shuffle
        self.preprocess=preprocess
        self.subset=subset
        self.info=info

        if self.subset == 'train':
                self.data_path='../input/train_images/'
        elif self.subset == 'test':
                self.data_path='../input/test_images/'
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        X=np.empty((self.batch_size,128,800,3),dtype=np.float32)
        y=np.empty((self.batch_size,128,800,4),dtype=np.int8)
        indexes=self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            X[i,]=Image.open(self.data_path+f).resize((800,128))
            if self.subset=='train':
                for j in range(4):
                    y[i,:,:,j]=rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])
        if self.preprocess != None:
            X=self.preprocess(X)
        if self.subset=='train':
            return X,y
        else:return X

    def on_epoch_end(self):
        self.indexes=np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    
