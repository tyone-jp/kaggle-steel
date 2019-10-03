from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from loss_function import dice_coef
import pandas as pd
from generator import DataGenerator
from keras.callbacks import ModelCheckpoint,CSVLogger

path='../output/model/'
train=pd.read_csv('../input/train2.csv')
train.fillna('',inplace=True)
train.reset_index(drop=True,inplace=True)

checkpoint=ModelCheckpoint(filepath=path+'unet.h5',monitor='val_dice_coef',save_best_only=True)
csv_logger=CSVLogger('../output/training.log')

if __name__=='__main__':
    preprocess=get_preprocessing('resnet34')
    model=Unet('resnet34',input_shape=(128,800,3),classes=4,activation='sigmoid')
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[dice_coef])

    idx=int(0.8*len(train))
    train_batches=DataGenerator(train.iloc[:idx],shuffle=True,preprocess=preprocess)
    valid_batches=DataGenerator(train.iloc[idx:],preprocess=preprocess)

    history=model.fit_generator(train_batches,validation_data=valid_batches,epochs=30,verbose=1,callbacks=[checkpoint,csv_logger])
