import os
import json
import gc

import cv2
import keras
from keras import backend as K
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,load_model
from keras.layers import Input
from keras.layers.covolutional import Conv2D,Conv2DTranspose
from keras.layrs.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback,ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from logging import StreamHandler,DEBUG,Formatter,FileHandler,getLogger

from generator import create_test_gen,DataGenerator
from mask2rle import build_rles,build_masks
from model import build_model

logger=getLogger(__name__)
DIR='../output/result/'

def load_img(code,base,resize=True):
    path=f'{base}/{code}'
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if resize:
        img=cv2.resize(img,(256,256))

    return img

def validate_path(path):
    if notos.path.exits(path):
        os.makedirs(path)
        
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
    logger.info('preprocessing')
    
    train_df=pd.read_csv('../input/train.csv')
    train_df['ImageId']=train_df['ImageId_ClassId'].apply(lambda x:x.split('_')[0])
    train_df['ClassId']=train_df['ImageId_ClassId'].apply(lambda x:x.split('_')[1])
    train_df['hasMask']=~train_df['EncodedPixels'].isna()

    logger.info('train_df shape:{}'.format(train_df.shape))

    mask_count_df=train_df.groupby('ImageId').agg(np.sum).reset_index()
    mask_count_df.sort_values('hasMask',ascending=False,inplace=True)

    logger.info('mask_count_df shape:{}'.format(mask_count_df.shape))

    sub_df=pd.read_csv('../input/sample_submission.csv')
    sub_df['ImageId']=sub_df['ImageId_ClassId'].apply(lambda x:x.split('_')[0])
    test_imgs=pd.DataFrame(sub_df['ImageId'].unique(),columns=['ImageId'])

    logger.info('test_imgs shape:{}'.format(test_imgs.shape))

    non_missing_train_idx=mask_count_df[mask_count_df['hasMask']>0]

    logger.info('non_missing_train_idx shape:{}'.format(non_missing_train_idx.shape))
    logger.info('remove test images without defects')

    test_gen=create_test_gen()

    remove_model=load_model('../output/model.h5')
    remove_model.summary()

    test_missing_pred=remove_model.predict_generator(
        test_gen,steps=len(test_gen),vebose=1)
    test_imgs['allMissing']=test_missing_pred

    logger.info('test_imgs',test_imgs.head())

    filtered_test_imgs=test_imgs[test_imgs['allMissing']<0.5]

    logger.info('filtered_test_imgs shape:{}'.format(filtered_test_imgs.shape))

    filtered_mask=sub_df['ImageId'].isin(filtered_test_imgs['ImageId'].values)
    filtered_sub_df=sub_df[filtered_mask].copy()
    null_sub_df=sub_df[~filtered_mask].copy()
    null_sub_df['EncodedPixels']=null_sub_df['EncodedPixels'].apply(lambda x:' ')
    filtered_sub_df.reset_index(drop=True,inplace=True)
    filtered_test_imgs.reset_index(drop=True,inplace=True)

    logger.info('filtered_sub_df shape:{}'.format(filtered_sub_df.shape))
    logger.info('filtered_test_imgs shape:{}'.format(filtered_test_imgs.shape))

    BATCH_SIZE=16
    train_idx,val_idx=train_test_split(non_missing_train_idx.index,random_state=2019,test_size=0.15)

    train_generator=DataGenerator(train_idx,df=mask_count_df,target_de=train_df,batch_size=BATCH_SIZE,n_classes=4)

    val_generator=DataGenerator(val_idx,df=mask_count_df,target_df=train_df,batch_size=BATCH_SIZE,n_classes=4)

    model=build_model((256,1600,1))
    model.summary()

    checkpoint=ModelCheckpoint('../output/model-unet.h5',monitor='val_loss',verbose=0,save_best_only=True,save_weights_only=False,mode='auto')

    history=model.fit_generator(train_generator,validation_data=val_generator,callbacks=[checkpoint],use_multiprocessing=False,workers=1,epochs=10)
    
    
                                                  
