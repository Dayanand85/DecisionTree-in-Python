# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 18:40:30 2022

@author: Dayanand
"""
# loading library

import os
import pandas as pd
import seaborn as sns
import numpy as np

# setting display size

pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",100)
pd.set_option("display.width",500)

# setting directory

os.getcwd()
os.chdir("C:\\Users\\tk\\Desktop\\DataScience\\Python Class Notes")

# loading dataset

fullRaw=pd.read_csv("Telecom_Churn.csv")
fullRaw.shape

# divide data into train & test

from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(fullRaw,train_size=.8,random_state=2410)
trainDf.shape
testDf.shape

# adding source column

trainDf["Source"]="Train"
testDf["Source"]="Test"

# concat both of the datasets

fullDf=pd.concat([trainDf,testDf],axis=0)
fullDf.shape

# summary of  full dataset

fullDf.describe()

# checking Null values

fullDf.isna().sum()

# drop idntifier columns

fullDf.columns
fullDf.drop(["customerID"],inplace=True,axis=1)
fullDf.shape

# Manualy convert dependent varaible levele

fullDf["Churn"]=np.where(fullDf["Churn"]=="Yes",1,0)
fullDf["Churn"].value_counts()
fullDf["Churn"].dtypes

# Outlier Detection & Correction
#fullDf.dtypes
#for i in fullDf.columns:
#   if(fullDf[i].dtypes!=object):
#        IQR3=fullDf[i]

#def Outlier_Detection_Correction(Df):
#     for i in Df.columns:
#         if(Df[i].dtypes!=object) & (i!="Churn"):
#             print(i)
#             Q1=np.percentile(Df[Df["Source"]=="Train",i],q=25)
#             Q3=np.percentile(Df[Df["Source"]=="Train",i],q=75)
#             IQR=Q3-Q1 
#             Lower_Bound=Q1-1.5*IQR
#             Upper_Bound=Q3+1.5*IQR
#             Df[:,i]=np.where(Df[:,i]<Lower_Bound,Lower_Bound,Df[:,i])
#             Df[:,i]=np.where(Df[:,i]>Upper_bound,Upper_Bound,Df[:,i])
#         return Df  

#fullDf2 = Outlier_Detection_Correction(fullDf)
#fullDf2

# boxplot

sns.boxplot(y=fullDf["TotalAmount"])

# Outlier Detection & Correction

columnsForOutlierDetection=["tenure","MonthlyServiceCharges","TotalAmount"]

summaryBeforeOutlierCorrection= fullDf.describe() 
for column in columnsForOutlierDetection:
    print(column)
    #Finding Q1 ,Q3 & IQR value 
    
    Q1=np.percentile(fullDf.loc[fullDf["Source"]=="Train",column],25)
    
    Q3=np.percentile(fullDf.loc[fullDf["Source"]=="Train",column],75)
    
    IQR=Q3-Q1

    
    # finding Lower_Bound & Upper_Bound value
    Lower_Bound=Q1-IQR*1.5
    Upper_Bound=Q3+IQR*1.5
    # Outlier operartion
    fullDf[column]=np.where(fullDf[column]<Lower_Bound,Lower_Bound,fullDf[column])
    fullDf[column]=np.where(fullDf[column]>Upper_Bound,Upper_Bound,fullDf[column])

summaryAfterOutlierCorrection=fullDf.describe()

# dummy variable creation
fullDf2 =pd.get_dummies(fullDf,drop_first=False)
fullDf2.shape

# Divide the data intot Train & Test

trainDf=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train"],axis=1).copy()
trainDf.shape
testDf=fullDf2[fullDf2["Source_Train"]==0].drop(["Source_Train"],axis=1).copy()
testDf.shape

# Divide the data into dep & indep variable

depVar="Churn"
trainX=trainDf.drop(depVar,axis=1).copy()
trainY=trainDf[depVar].copy()
testX=testDf.drop(depVar,axis=1).copy()
testY=testDf[depVar].copy()

trainX.shape
testX.shape
trainY.shape
testY.shape

# Decision Tree building
from sklearn.tree import DecisionTreeClassifier ,plot_tree
from matplotlib.pyplot import figure,savefig,close

M1=DecisionTreeClassifier(random_state=2410).fit(trainX,trainY)

 # Model Visualization
 
figure(figsize=[10,15])

DT_plot=plot_tree(M1,fontsize=10,feature_names=trainX.columns,filled=True,class_names=["0","1"])

# Prediction & validation on testset

from sklearn.metrics import classification_report

# prediction on testset

Test_predict=M1.predict(testX)

# confusion matrix
conf_mat=pd.crosstab(testY,Test_predict) # Actual,predicted
conf_mat

# classification_report
print(classification_report(testY,Test_predict))

# DT Model 2 with tuning branches

M2=DecisionTreeClassifier(random_state=2410,min_samples_leaf=500).fit(trainX,trainY)

DT_plot2=plot_tree(M2,fontsize=10,feature_names=trainX.columns,filled=True,class_names=["0","1"])

# predict on test set with new model

test_predict1=M2.predict(testX)

# confusion_matrix

conf_mat2=pd.crosstab(testY,test_predict1)

# classification report
print(classification_report(testY,test_predict1))
