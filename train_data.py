import pandas as pd
import numpy as np

train_df=pd.read_csv('../input/train.csv')

if __name__=='__main__':
    train_df['ImageId']=train_df['ImageId_ClassId'].map(lambda x:x.split('_')[0])
    train2_df=pd.DataFrame({'ImageId':train_df['ImageId'][::4]})
    train2_df['e1']=train_df['EncodedPixels'][::4].values
    train2_df['e2']=train_df['EncodedPixels'][1::4].values
    train2_df['e3']=train_df['EncodedPixels'][2::4].values
    train2_df['e4']=train_df['EncodedPixels'][3::4].values
    train2_df.reset_index(inplace=True,drop=True)
    train2_df.fillna('',inplace=True)

    train2_df.to_csv('../input/train2.csv')
    
