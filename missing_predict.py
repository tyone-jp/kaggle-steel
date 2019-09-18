import os
import json

import cv2
import numpy as np
import pandas as pd
import keras
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback,ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam,Nadam
import tensorflow as tf
from tqdm import tqdm

from logging import StreamHandler,DEBUG,Formatter,FileHandler,getLogger

logger=getLogger(__name__)
DIR='../result/tmp/'

def load_img(code,base,resize=True):
    path=f'{base}/{code}'
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if resize:
        img=cv2.resize(img,(256,256))
        
    return img

def create_datagen():
    return ImageDataGenerator(zoom_range=0.1,
                              fill_mode='constant',
                              cval=0.,
                              rotation_range=10,
                              height_shift_range=0.1,
                              width_shift_range=0.1,
                              horizontal_flip=True,
                              vertical_flip=True,
                              rescale=1/255.,
                              validation_split=0.15)

def create_test_gen():
    return ImageDataGenerator(rescale=1/255.).flow_from_dataframe(test_nan_df,
                                                                  directory='../input/test_images/',
                                                                  x_col='ImageId',
                                                                  class_mode=None,
                                                                  target_size=(256,256),
                                                                  batch_size=BATCH_SIZE,
                                                                  shuffle=False)
def create_flow(datagen,subset):
    return datagen.flow_from_dataframe(train_nan_df,
                                       directory='../tmp/train',
                                       x_col='ImageId',
                                       y_col='allMissing',
                                       class_mode='other',
                                       target_size=(256,256),
                                       batch_size=BATCH_SIZE,
                                       subset=subset)

def build_model():
    densenet=DenseNet121(include_top=False,
                         input_shape=(256,256,3),
                         weights='../input/weight/DenseNet-BC-121-32-no-top.h5')

    model=Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=['accuracy'])
    
    return model

def tta_prediction(datagen,model,image,n_examples):
    samples=np.expand_dims(image,axis=0)
    it=datagen.flow(samples,batch_size=n_examples)
    yhats=model.predict_generator(it,steps=n_examples,verbose=0)
    summed=np.sum(yhats,axis=0)/n_examples
    return summed

if __name__=='__main__':
    log_fmt=Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s] [%(funcName)s] %(message)s')
    handler=StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)
    
    handler=FileHandler(DIR+'train.py.log','a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    
    logger.info('start')

    train_df=pd.read_csv('../input/train.csv')
    submission_df=pd.read_csv('../input/sample_submission.csv')

    logger.info('train_df shape:{}'.format(train_df.shape))
    logger.info('submisson_df shape:{}'.format(submission_df.shape))
                
    unique_test_images=submission_df['ImageId_ClassId'].apply(lambda x:x.split('_')[0]).unique()
    train_df['isNan']=pd.isna(train_df['EncodedPixels'])
    train_df['ImageId']=train_df['ImageId_ClassId'].apply(lambda x:x.split('_')[0])

    train_nan_df=train_df.groupby(by='ImageId',axis=0).agg('sum')
    train_nan_df.reset_index(inplace=True)
    train_nan_df.rename(columns={'isNan':'missingCount'},inplace=True)
    train_nan_df['missingCount']=train_nan_df['missingCount'].astype(np.int32)
    train_nan_df['allMissing']=(train_nan_df['missingCount']==4).astype(int)
    train_nan_df['ImageId']=train_nan_df['ImageId'].apply(lambda x:x.replace('.jpg','.png'))
    
    logger.info('train_nan_df shape:{}'.format(train_nan_df.shape))

    test_nan_df=pd.DataFrame(unique_test_images,columns=['ImageId'])

    logger.info('test_nan_df shape:{}'.format(test_nan_df.shape))
    logger.info('data download finished')
    logger.info('train start')

    logger.info('create data_generator')
    BATCH_SIZE=32
    data_generator=create_datagen()
    train_gen=create_flow(data_generator,'training')
    val_gen=create_flow(data_generator,'validation')
    test_gen=create_test_gen()

    model=build_model()

    logger.info('train')

    total_steps=train_nan_df.shape[0]/BATCH_SIZE

    checkpoint=ModelCheckpoint('../output/model.h5',
                               monitor='val_acc',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=False,
                               mode='auto')

    reduce_lr=ReduceLROnPlateau(monitor='val_loss',
                                patience=5,
                                verbose=1,
                                min_lr=1e-6)

    history=model.fit_generator(train_gen,
                                steps_per_epoch=total_steps*0.85,
                                validation_data=val_gen,
                                validation_steps=total_steps*0.15,
                                epochs=30,
                                callbacks=[checkpoint,reduce_lr])

    logger.info('train finished')

    history_df=pd.DataFrame(history.history)
    history_df.to_csv('../output/history.csv')

    logger.info('history saved')
    
    model.load_weights('../output/model.h5')
    y_test=np.empty(test_nan_df.shape)
    for i,code in enumerate(tqdm(test_nan_df['ImageId'])):
            y_test[i]=tta_prediction(datagen=create_datagen(),
                                     model=model,
                                     image=load_img(base='../input/test_images',code=code),
                                     n_examples=20)

    logger.info('tta finished')

    test_nan_df['allMissing']=y_test

    train_nan_df.to_csv('train_missing_count.csv',index=False)
    test_nan_df.to_csv('test_missing_count.csv',index=False)

    logger.info('finish')
            
