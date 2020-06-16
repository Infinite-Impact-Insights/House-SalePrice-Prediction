'''
Created on Jun 1, 2020

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
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

trainDataset = pd.read_csv('train.csv')
testDataset = pd.read_csv('test.csv')

train_df=pd.DataFrame(trainDataset)
test_df=pd.DataFrame(testDataset)

train_df.drop(['SalePrice'],axis=1, inplace=True)


# for column in wholeDataset.columns:
#     wholeDataset[column].fillna(wholeDataset[column].mode()[0], inplace=True)

for column in train_df.columns:
    train_df[column].fillna(train_df[column].mode()[0], inplace=True)
for column in test_df.columns:
    test_df[column].fillna(test_df[column].mode()[0], inplace=True)
wholeDataset=pd.concat([train_df,test_df],ignore_index=True)
     
     
title=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
       'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterQual','ExterCond','Foundation','Heating',
       'HeatingQC','CentralAir','KitchenQual','Functional','PavedDrive','SaleType','SaleCondition','MasVnrType','BsmtQual','BsmtCond',
       'BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
       'PoolQC','Fence','MiscFeature','Alley']
 
     
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


# for f in title:
#     train_df['MS int']=labelencoder.fit_transform(train_df[f])
#     enc = OneHotEncoder(handle_unknown='ignore')
#     enc_df=pd.DataFrame(enc.fit_transform(train_df[['MS int']]).toarray())
#     train_df= train_df.join(enc_df)
#     max=train_df['MS int'].max()
#     maxplus=max+1
#     train_df.drop([f,'MS int'], axis=1, inplace=True)
#     for i in range(0,int(maxplus)):
#         train_df.rename(columns={i:str(f)+" "+str(i)},inplace=True)
#          
# for t in title:
#     test_df['MS int']=labelencoder.fit_transform(test_df[t])
#     enc = OneHotEncoder(handle_unknown='ignore')
#     enc_df=pd.DataFrame(enc.fit_transform(test_df[['MS int']]).toarray())
#     test_df= test_df.join(enc_df)
#     testmax=test_df['MS int'].max()
#     testmaxplus=testmax+1
#     test_df.drop([t,'MS int'], axis=1, inplace=True)
#     for i in range(0,int(testmaxplus)):
#         test_df.rename(columns={i:str(t)+" "+str(i)},inplace=True)
 
# train_df=train_df.join(trainDataset['SalePrice'])
        
print(train_df.shape)
print(test_df.shape)
print(wholeDataset)

scoreData=[[0,0,0,0,0]]
scoreArr=pd.DataFrame(columns=['Max Features','Max depth','No. of Features','Training Score', 'Testing Score'])

dataset=wholeDataset.values
print(dataset[1460,0])
X = dataset[0:1460,2:289]
Y = trainDataset['SalePrice']
for feat in range(10,289,10):
    for dept in range(5,feat,10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        classifier = RandomForestRegressor(n_estimators = 1000, max_features=feat,max_depth=dept)
        classifier.fit(X_train, Y_train)
        scoreArr=scoreArr.append({'Max Features':feat,'Max depth':dept,'Training Score':classifier.score(X, Y),'Testing Score':classifier.score(X_test, Y_test)},ignore_index=True)
        print(classifier.score(X_train, Y_train)," : ",feat)
     
scoreArr.to_csv('allScore3.csv')    
#test_df.to_csv("lessee2.csv",sep=',')
# train_df.to_csv("trainls2.csv",sep=',')

# dataset=wholeDataset.values
# print(dataset.shape)
# 
# X = dataset[0:1460,2:289]
# Y = trainDataset['SalePrice']
# classifier = RandomForestRegressor(n_estimators = 1000, max_features=220)
# classifier.fit(X, Y)
# print("Score",classifier.score(X, Y))
# res=classifier.predict(dataset[1460:,2:289])
# print(res)

# testset=test_df.values
# res=classifier.predict(testset[:,2:])
# print(res)
# pred=pd.DataFrame(res)
# pred.to_csv('prediction_results2.csv',sep=',')


