'''
Created on Jun 3, 2020

@author: gaurav
'''
from time import time
from sklearn import metrics
from json.decoder import NaN
from math import nan
from sklearn.ensemble import RandomForestRegressor
from unittest.mock import inplace
from pandas.core.dtypes.missing import isnull
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as AS
import numpy as np
import pandas as pd
from numpy import array
import xgboost as xgb
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from xgboost.core import Booster

trainDataset = pd.read_csv('train.csv')
testDataset = pd.read_csv('test.csv')

train_df=pd.DataFrame(trainDataset)
test_df=pd.DataFrame(testDataset)

train_df.drop(['SalePrice'],axis=1, inplace=True)

for column in train_df.columns:
    train_df[column].fillna(train_df[column].mode()[0], inplace=True)
for column in test_df.columns:
    test_df[column].fillna(test_df[column].mode()[0], inplace=True)
wholeDataset=pd.concat([train_df,test_df],ignore_index=True)
wholeDataset.drop(['PoolQC'],axis=1, inplace=True)
wholeDataset.drop(['Fence'],axis=1, inplace=True)
wholeDataset.drop(['Alley'],axis=1, inplace=True)
wholeDataset.drop(['MiscFeature'],axis=1, inplace=True)

     
title=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
       'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterQual','ExterCond','Foundation','Heating',
       'HeatingQC','CentralAir','KitchenQual','Functional','PavedDrive','SaleType','SaleCondition','MasVnrType','BsmtQual','BsmtCond',
       'BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']
 

   
labelencoder = LabelEncoder()
for tit in title:
    wholeDataset['MS int']=labelencoder.fit_transform(wholeDataset[tit])
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df=pd.DataFrame(enc.fit_transform(wholeDataset[['MS int']]).toarray())
    wholeDataset= wholeDataset.join(enc_df)
    max=wholeDataset['MS int'].max()
    maxplus=max+1
    wholeDataset.drop([tit,'MS int'], axis=1, inplace=True)
    for i in range(0,int(maxplus)):
        wholeDataset.rename(columns={i:str(tit)+" "+str(i)},inplace=True)
        
print(train_df.shape)
print(test_df.shape)
print(wholeDataset)

scoreData=[[0,0,0,0,0]]
scoreArr=pd.DataFrame(columns=['Max Features','Max depth','No. of Features','Training Score', 'Testing Score'])

dataset=wholeDataset.values
print(dataset[1460,0])
X = dataset[0:1460,1:276]
Y = trainDataset['SalePrice']
for feat in range(10,276,10):
    cscore=0
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    classifier = xgb.XGBRegressor(learning_rate=0.01,
                   n_estimators=6000,
                   max_depth=4,
                   min_child_weight=0,
                   gamma=0.6,
                   subsample=0.7,
                   colsample_bytree=0.7,
#                       objective='reg:linear',
                   scale_pos_weight=1,
                   seed=27,
                   reg_alpha=0.00006,
                   random_state=42)
    classifier.fit(X_train, Y_train)
    cscore=cscore + classifier.score(X_test, Y_test)
    print(classifier.score(X_test, Y_test)," : ",feat)
    scoreArr=scoreArr.append({'Max Features':feat,'Max depth':'null','Training Score':classifier.score(X_train, Y_train),'Testing Score':(cscore/5)},ignore_index=True)
      
scoreArr.to_csv('allScore6.csv')    
# test_df.to_csv("lessee2.csv",sep=',')
# train_df.to_csv("trainls2.csv",sep=',')

# dataset=wholeDataset.values
#  
# print(dataset.shape)
#   
# X = dataset[0:1460,2:276]
# Y = trainDataset['SalePrice']
# classifier = xgb.XGBRegressor(colsample_bytree=0.7,
#                  gamma=0,
#                  min_child_weight=1.5,
#                  n_estimators=10000,                                                                    
#                  reg_alpha=0.75,
#                  reg_lambda=0.45,
#                  subsample=0.6,
#                  seed=42,
#                  max_features=210)
#           
# classifier.fit(X, Y)
# print("Score",classifier.score(X, Y))
# res=classifier.predict(dataset[1460:,2:276])
# print(res)

# testset=test_df.values
# res=classifier.predict(testset[:,2:])
# print(res)
# pred=pd.DataFrame(res)
# pred.to_csv('prediction_results4.csv',sep=',')


