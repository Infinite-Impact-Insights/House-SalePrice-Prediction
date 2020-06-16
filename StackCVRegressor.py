'''
Created on Jun 7, 2020

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
from sklearn import ensemble
import lightgbm as lgb
from mlxtend.regressor import StackingCVRegressor

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
        
        
print("start")
dataset=wholeDataset.values  
print(dataset.shape)
X = dataset[0:1460,2:276]
Y = trainDataset['SalePrice']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
xgbclassifier = xgb.XGBRegressor(colsample_bytree=0.7,
                 gamma=0,
                 min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42,
                 max_features=220)
xgbclassifier.fit(X_train, Y_train)

rfclassifier = RandomForestRegressor(n_estimators = 1000, max_features=220)
rfclassifier.fit(X_train, Y_train)

stack_gen = StackingCVRegressor(regressors=(xgbclassifier,rfclassifier),
                                meta_regressor=xgbclassifier,
                                use_features_in_secondary=True)
stack_gen.fit(X_train,Y_train)


print("Score rf",rfclassifier.score(X_test, Y_test))
print("Score xgb",xgbclassifier.score(X_test, Y_test))
print("Score stack",stack_gen.score(X_test, Y_test))
