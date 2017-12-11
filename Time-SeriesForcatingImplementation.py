# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#------------This script is for creating window features------------------ 
#This script extracts the ids for a given period
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import datetime as dt
#import tensorflow as tf
#export TF_CPP_MIN_LOG_LEVEL=2
#------------------------This script extracts target values------------------------------

#Reading source file & sort values by parcelid
source=pd.read_csv('../input/zillow-prize-1/train_2016_v2.csv',low_memory=False,parse_dates=['transactiondate'])
source=source.sort_values('parcelid')

#dates=['2016-04-01','2016-05-01','2016-06-01', '2016-07-01']

start=['2016-04-01','2016-05-01','2016-06-01']
end=['2016-05-01','2016-06-01', '2016-07-01']

#Extracting ids based on specified time periods for each datset
def extractingTargets(startDate,endDate):
    for i in range(len(startDate)):
        extractedIds=source[(source.transactiondate >= startDate[i]) & (source.transactiondate< endDate[i])]

        def fileName():
            yr=startDate[i].split('-')[0]
            for j in range(12):
                if (startDate[i].split('-')[1]=='0'+str(j)):
                    return '0'+str(j)+'-'+yr+'.csv'
                else: print ('Month not available')
       
        fn=fileName()

        #eliminating duplicates and saving extracted results
        extractedIds.drop_duplicates(['parcelid'],keep='last').to_csv(fn,header=True,index=False)
    
extractingTargets(start,end)

windowStartApr=['2016-01-01','2016-02-01','2016-03-01']
windowEndApr=['2016-04-01','2016-04-01','2016-04-01']

windowStartMay=['2016-02-01','2016-03-01','2016-04-01']
windowEndMay=['2016-05-01','2016-05-01','2016-05-01']

windowStartJun=['2016-03-01','2016-04-01','2016-05-01']
windowEndJun=['2016-06-01','2016-06-01','2016-06-01']


def getfile(file):
    Ids=pd.read_csv(file)
    Ids=Ids.set_index('parcelid')
    Ids=Ids.drop(['logerror'],axis=1)
    Ids=Ids.reset_index()
    return Ids
    
IdsApr=getfile('04-2016.csv')
IdsMay=getfile('05-2016.csv')
IdsJun=getfile('06-2016.csv')

print (IdsApr.head())

#stat measures on variable windows
def slidingWindow(windowStart,windowEnd):
    for i in range(len(windowStart)):
        
        def computingStatFeatures(VW):
            df0=VW.groupby('parcelid').mean()
            df1=VW.groupby('parcelid').median()
            df2=VW.groupby('parcelid').std()
            df3=VW.groupby('parcelid').min()
            df4=VW.groupby('parcelid').max()
            df5=VW.groupby('parcelid').quantile(.75)
            dfc=pd.concat([df0,df1,df2,df3,df4,df5],axis=1)
            dfc=pd.DataFrame(dfc)
            print ("Reseult"+str(dfc))
            return dfc
        
        def Window():
                variableWindow=source[(source.transactiondate >= windowStart[i]) & (source.transactiondate< windowEnd[i])] 
                #variableWindow=pd.DataFrame(variableWindow)
                #variableWindow['Month']=variableWindow['transactiondate'].dt.month
                #variableWindow['Day']=variableWindow['transactiondate'].dt.day
                variableWindow=variableWindow.drop(['transactiondate'],axis=1)
                return variableWindow
        
        def retVarThree():
                    ThreeMonthWindow=Window()
                    ThreemEF=computingStatFeatures(ThreeMonthWindow)
                    ThreemEF=ThreemEF.reset_index()
                    return ThreemEF
        def retVarTwo():
                    TwoMonthWindow=Window()
                    TwomEF=computingStatFeatures(TwoMonthWindow)
                    TwomEF=TwomEF.reset_index()
                    return TwomEF
        def retVarOne():
                    OneMonthWindow=Window()
                    OnemEF=computingStatFeatures(OneMonthWindow)
                    OnemEF=OnemEF.reset_index()
                    return OnemEF
                    
        Targets=[IdsApr,IdsMay,IdsJun] 
        m=4
        for k in range(len(Targets)):
            if (int(windowEnd[i].split('-')[1])-int(windowStart[i].split('-')[1])==3 and int(windowEnd[i].split('-')[1])==m):
                    ThreeMonthWindow=retVarThree()
                    res1=pd.merge(ThreeMonthWindow,Targets[k],on='parcelid').drop(['transactiondate'],axis=1)
                    print (res1)
    
            if (int(windowEnd[i].split('-')[1])-int(windowStart[i].split('-')[1])==2 and int(windowEnd[i].split('-')[1])==m):
                    TwoMonthWindow=retVarTwo()
                    res2=pd.merge(TwoMonthWindow,Targets[k],on='parcelid').drop(['transactiondate'],axis=1)
                    print (res2)
    
            if (int(windowEnd[i].split('-')[1])-int(windowStart[i].split('-')[1])==1 and int(windowEnd[i].split('-')[1])==m):
                    OneMonthWindow=retVarOne()
                    res3=pd.merge(OneMonthWindow,Targets[k],on='parcelid').drop(['transactiondate'],axis=1)
                    print (res3)
                    pd.DataFrame(pd.concat([res1,res2.drop(['parcelid'],axis=1,inplace=True),res3.drop(['parcelid'],axis=1,inplace=True)],axis=1)).to_csv('EngineeredFeatures'+str(k)+'.csv',index=False,header=['parcelid','mean','median','std','min','max','quantile'])
            m=m+1   
             
slidingWindow(windowStartApr,windowEndApr) 
slidingWindow(windowStartMay,windowEndMay) 
slidingWindow(windowStartJun,windowEndJun) 


#-------------This script extracts features-----------------------------------------------
import pandas as pd

IdsApr=IdsApr.set_index('parcelid')
IdsMay=IdsMay.set_index('parcelid')
IdsJun=IdsJun.set_index('parcelid')
Targets=[IdsApr,IdsMay,IdsJun] 

def extractFeatures():
    for e in range(len(Targets)):
        data_file = pd.read_csv('../input/zillow-prize-1/properties_2016.csv',low_memory=False)
        data_file=data_file.sort_values('parcelid')
        data_file=data_file.set_index('parcelid')
        res = data_file.loc[Targets[e].index]
        print (IdsApr.head())
        print (data_file.head())
        res.to_csv('ExtractedStaticFeatures'+str(e)+'.csv')

extractFeatures()


#----------------------Merging features targets------------------------------------

import pandas as pd
files=['04-2016.csv','05-2016.csv','06-2016.csv']
for l in range (len(files)):
    df1=pd.read_csv('ExtractedStaticFeatures'+str(l)+'.csv')
    df2=pd.read_csv('EngineeredFeatures'+str(l)+'.csv')
    df3=pd.read_csv(files[l])
    print(df3.head())
    #df3=pd.read_csv('../input/targetdata/Targets-Apr.csv')
    
    df2=df2.drop('parcelid',axis=1)
    df3=df3.drop(['parcelid','transactiondate'],axis=1)
    
    df4=pd.concat([df1,df2,df3],axis=1)
    
    df4=df4 [[u'parcelid', u'airconditioningtypeid', u'architecturalstyletypeid',
           u'basementsqft', u'bathroomcnt', u'bedroomcnt', u'buildingclasstypeid',
           u'buildingqualitytypeid', u'calculatedbathnbr', u'decktypeid',
           u'finishedfloor1squarefeet', u'calculatedfinishedsquarefeet',
           u'finishedsquarefeet12', u'finishedsquarefeet13',
           u'finishedsquarefeet15', u'finishedsquarefeet50',
           u'finishedsquarefeet6', u'fips', u'fireplacecnt', u'fullbathcnt',
           u'garagecarcnt', u'garagetotalsqft', u'hashottuborspa',
           u'heatingorsystemtypeid', u'latitude', u'longitude',
           u'lotsizesquarefeet', u'poolcnt', u'poolsizesum', u'pooltypeid10',
           u'pooltypeid2', u'pooltypeid7', u'propertycountylandusecode',
           u'propertylandusetypeid', u'propertyzoningdesc',
           u'rawcensustractandblock', u'regionidcity', u'regionidcounty',
           u'regionidneighborhood', u'regionidzip', u'roomcnt', u'storytypeid',
           u'threequarterbathnbr', u'typeconstructiontypeid', u'unitcnt',
           u'yardbuildingsqft17', u'yardbuildingsqft26', u'yearbuilt',
           u'numberofstories', u'fireplaceflag', u'structuretaxvaluedollarcnt',
           u'taxvaluedollarcnt', u'assessmentyear', u'landtaxvaluedollarcnt',
           u'taxamount', u'taxdelinquencyflag', u'taxdelinquencyyear',
           u'censustractandblock', u'mean',u'median',u'std',u'min',u'max',u'quantile',u'logerror']]
    
    df4.to_csv('Dataset'+str(l)+'.csv',index=False)


#----------------------Handling Categorical values-----------------------------

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

replacements=['Dataset0.csv','Dataset1.csv','Dataset2.csv']
#this script converts categorical variables to numeric

def HandleCatValues():
    for i in range (len(replacements)):
        df1=pd.read_csv(filepath_or_buffer=replacements[i])
        ext=pd.DataFrame(df1,columns=['fireplaceflag','propertycountylandusecode','propertyzoningdesc','hashottuborspa','taxdelinquencyflag'])
        
        le = LabelEncoder()
        ext['fireplaceflag']=le.fit_transform(ext['fireplaceflag'].astype('str'))
        ext['hashottuborspa']=le.fit_transform(ext['hashottuborspa'].astype('str'))
        ext['propertycountylandusecode']=le.fit_transform(ext['propertycountylandusecode'].astype('str'))
        ext['propertyzoningdesc']=le.fit_transform(ext['propertyzoningdesc'].astype('str'))
        ext['taxdelinquencyflag']=le.fit_transform(ext['taxdelinquencyflag'].astype('str'))
        
        #print ext
        
        df1=df1.drop(['fireplaceflag','propertycountylandusecode','propertyzoningdesc','hashottuborspa','taxdelinquencyflag'],axis=1)
        df1=pd.concat([df1,ext],axis=1)
    
        #Handling some categorical features according to mentor
        
        df6=df1.loc[:,['propertyzoningdesc','logerror']]
        df2=df1.loc[:,['propertycountylandusecode','logerror']]
        
        df6=df6.groupby('propertyzoningdesc').mean()
        df2=df2.groupby('propertycountylandusecode').mean()
        
        df3=df1['propertyzoningdesc']
        df4=df1['propertycountylandusecode']
        
        df6=df6.reset_index('propertyzoningdesc')
        df2=df2.reset_index('propertycountylandusecode')
        
        
        res1 = df6.loc[df3.index]
        res2= df2.loc[df4.index]
        
        
        #res1=res1.reset_index('propertyzoningdesc')
        #res2=res2.reset_index('propertycountylandusecode')
        
        res1=res1.drop('propertyzoningdesc',axis=1)
        res2=res2.drop('propertycountylandusecode',axis=1)
        
        res1=res1.rename(columns={'logerror':'propertyzoningdesc'})
        res2=res2.rename(columns={'logerror':'propertycountylandusecode'})
        
        df5=pd.concat([res1,res2],axis=1)
        df5=pd.DataFrame(df5)
        
        df1=df1.drop(['propertyzoningdesc','propertycountylandusecode'],axis=1)
        df1=pd.concat([df1,df5],axis=1)
        
        df10=df1 [[u'parcelid', u'airconditioningtypeid', u'architecturalstyletypeid',
               u'basementsqft', u'bathroomcnt', u'bedroomcnt', u'buildingclasstypeid',
               u'buildingqualitytypeid', u'calculatedbathnbr', u'decktypeid',
               u'finishedfloor1squarefeet', u'calculatedfinishedsquarefeet',
               u'finishedsquarefeet12', u'finishedsquarefeet13',
               u'finishedsquarefeet15', u'finishedsquarefeet50',
               u'finishedsquarefeet6', u'fips', u'fireplacecnt', u'fullbathcnt',
               u'garagecarcnt', u'garagetotalsqft', u'hashottuborspa',
               u'heatingorsystemtypeid', u'latitude', u'longitude',
               u'lotsizesquarefeet', u'poolcnt', u'poolsizesum', u'pooltypeid10',
               u'pooltypeid2', u'pooltypeid7', u'propertycountylandusecode',
               u'propertylandusetypeid', u'propertyzoningdesc',
               u'rawcensustractandblock', u'regionidcity', u'regionidcounty',
               u'regionidneighborhood', u'regionidzip', u'roomcnt', u'storytypeid',
               u'threequarterbathnbr', u'typeconstructiontypeid', u'unitcnt',
               u'yardbuildingsqft17', u'yardbuildingsqft26', u'yearbuilt',
               u'numberofstories', u'fireplaceflag', u'structuretaxvaluedollarcnt',
               u'taxvaluedollarcnt', u'assessmentyear', u'landtaxvaluedollarcnt',
               u'taxamount', u'taxdelinquencyflag', u'taxdelinquencyyear',
               u'censustractandblock', u'mean',u'median',u'std',u'min',u'max',u'quantile',u'logerror']]
        
        print (df10.shape)
        
        df10.to_csv('FinalDataset'+str(i)+'.csv',index=False)
            
HandleCatValues()

#-----------------Appending Datasets to create entire datset---------------------------------
import pandas as pd

df1=pd.read_csv('FinalDataset0.csv')
df2=pd.read_csv('FinalDataset1.csv')
df3=pd.read_csv('FinalDataset2.csv')

df4=df1.append(df2)
df5=df4.append(df3)

df5.to_csv('Trainset.csv',index=False)

#-----------FeatureEngineering-------------------------
NF=pd.read_csv('Trainset.csv')
NF['NullCount']=NF.isnull().sum(axis=1)
NF['location']=NF['longitude']*NF['latitude']
NF['RoomCount']=NF['bathroomcnt']+NF['bedroomcnt']
NF['AgeOfProperty']=2016-NF['yearbuilt']


NF=NF[[u'parcelid', u'airconditioningtypeid', u'architecturalstyletypeid',
               u'basementsqft', u'bathroomcnt', u'bedroomcnt', u'buildingclasstypeid',
               u'buildingqualitytypeid', u'calculatedbathnbr', u'decktypeid',
               u'finishedfloor1squarefeet', u'calculatedfinishedsquarefeet',
               u'finishedsquarefeet12', u'finishedsquarefeet13',
               u'finishedsquarefeet15', u'finishedsquarefeet50',
               u'finishedsquarefeet6', u'fips', u'fireplacecnt', u'fullbathcnt',
               u'garagecarcnt', u'garagetotalsqft', u'hashottuborspa',
               u'heatingorsystemtypeid', u'latitude', u'longitude',
               u'lotsizesquarefeet', u'poolcnt', u'poolsizesum', u'pooltypeid10',
               u'pooltypeid2', u'pooltypeid7', u'propertycountylandusecode',
               u'propertylandusetypeid', u'propertyzoningdesc',
               u'rawcensustractandblock', u'regionidcity', u'regionidcounty',
               u'regionidneighborhood', u'regionidzip', u'roomcnt', u'storytypeid',
               u'threequarterbathnbr', u'typeconstructiontypeid', u'unitcnt',
               u'yardbuildingsqft17', u'yardbuildingsqft26', u'yearbuilt',
               u'numberofstories', u'fireplaceflag', u'structuretaxvaluedollarcnt',
               u'taxvaluedollarcnt', u'assessmentyear', u'landtaxvaluedollarcnt',
               u'taxamount', u'taxdelinquencyflag', u'taxdelinquencyyear',
               u'censustractandblock', u'mean',u'median',u'std',u'min',u'max',u'quantile',u'NullCount',
               u'location',u'RoomCount',u'AgeOfProperty',u'logerror']]
               
arr=[u'parcelid', u'airconditioningtypeid', u'architecturalstyletypeid',
               u'basementsqft', u'bathroomcnt', u'bedroomcnt', u'buildingclasstypeid',
               u'buildingqualitytypeid', u'calculatedbathnbr', u'decktypeid',
               u'finishedfloor1squarefeet', u'calculatedfinishedsquarefeet',
               u'finishedsquarefeet12', u'finishedsquarefeet13',
               u'finishedsquarefeet15', u'finishedsquarefeet50',
               u'finishedsquarefeet6', u'fips', u'fireplacecnt', u'fullbathcnt',
               u'garagecarcnt', u'garagetotalsqft', u'hashottuborspa',
               u'heatingorsystemtypeid', u'latitude', u'longitude',
               u'lotsizesquarefeet', u'poolcnt', u'poolsizesum', u'pooltypeid10',
               u'pooltypeid2', u'pooltypeid7', u'propertycountylandusecode',
               u'propertylandusetypeid', u'propertyzoningdesc',
               u'rawcensustractandblock', u'regionidcity', u'regionidcounty',
               u'regionidneighborhood', u'regionidzip', u'roomcnt', u'storytypeid',
               u'threequarterbathnbr', u'typeconstructiontypeid', u'unitcnt',
               u'yardbuildingsqft17', u'yardbuildingsqft26', u'yearbuilt',
               u'numberofstories', u'fireplaceflag', u'structuretaxvaluedollarcnt',
               u'taxvaluedollarcnt', u'assessmentyear', u'landtaxvaluedollarcnt',
               u'taxamount', u'taxdelinquencyflag', u'taxdelinquencyyear',
               u'censustractandblock', u'mean',u'median',u'std',u'min',u'max',u'quantile',u'NullCount',
               u'location',u'RoomCount',u'AgeOfProperty',u'logerror']


for i in range(len(arr)):  
    y=NF[arr[i]].nunique()
    
    if(y==1):
        #print("x"+str(x))
        print("y:"+str(y))
        NF=NF.drop(arr[i],axis=1)
        
arr2=NF.keys()
for i in range(len(arr2)):  
    x=(((NF[arr2[i]].isnull().sum())/NF.shape[0])*100)

    if(x>90):
        #print("x"+str(x))
        print("x:"+str(x))
        NF=NF.drop(arr2[i],axis=1)
        NF.to_csv('Trainset.csv',index=False)

#-----------fillling missing values--------------------
df6=pd.read_csv('Trainset.csv')
df6=df6.fillna(404)
print (df6.shape)
df6.to_csv('Trainset.csv',index=False)
#---------------------------XGB model-------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
import joblib

TrainSet=np.loadtxt('Trainset.csv',delimiter=',',skiprows=1)

features = TrainSet[:,1:40]
labels = TrainSet[:,40]

tr_features, ts_features, tr_labels, ts_labels = train_test_split(features,labels, test_size=0.30, random_state=42)

#imputer=Imputer(missing_values=404,strategy='median')
#tr_features=imputer.fit_transform(tr_features)
#ts_features=imputer.transform(ts_features)

data=xgb.DMatrix(data=tr_features,label=tr_labels)
test=xgb.DMatrix(data=ts_features)

eval_set=xgb.DMatrix(ts_features,ts_labels)

params={'booster':'gbtree','eta':0.1,'seed':0,'subsample':1,'colsample_bytree':1,'objective':'reg:linear','max_depth':3,'min_child_weight':2,'silent':1,'eval_metric':'mae','verbose':True,'gamma':0}
bst=xgb.train(params,data,num_boost_round=7000,evals=[(eval_set,'eval')],early_stopping_rounds=10)

y_pred=bst.predict(test)

res=xgb.cv(params = params, dtrain = data, num_boost_round = 7000, nfold = 10, metrics = ['mae'],early_stopping_rounds = 10)

print (res)

print ("MAE:XGBoost",mean_absolute_error(ts_labels,y_pred))
joblib.dump(bst, "xgboost.joblib.dat")

#keras neural net
import keras as ks

# define base model
import numpy
import pandas as pd
from keras.layers import Dense, Dropout

from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
import numpy as np

TrainSet= pd.np.loadtxt('Trainset.csv', delimiter=',',skiprows=1)

print (TrainSet.shape)
features = TrainSet[:,1:40]
labels = TrainSet[:,40]

tr_features, ts_features, tr_labels, ts_labels = train_test_split(features,labels, test_size=0.30, random_state=42)

#imputer=Imputer(missing_values=-1,strategy='most_frequent')
#tr_features=imputer.fit_transform(tr_features)
#ts_features=imputer.transform(ts_features)

tr_labels=tr_labels.reshape(-1,1)
ts_labels=ts_labels.reshape(-1,1)

scale_features=StandardScaler()
scale_labels=StandardScaler()

tr_features=scale_features.fit_transform(tr_features)
ts_features=scale_features.transform(ts_features)

tr_labels=scale_labels.fit_transform(tr_labels)
ts_labels=scale_labels.transform(ts_labels)

# create model
def constructModel():
    model = Sequential()
    model.add(Dense(39, input_dim=39, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    # Compile model
    adam=Adam(lr=0.01)
    model.compile(loss='mae',optimizer=adam ,metrics=['mae'])
    return model



kr=KerasRegressor(build_fn=constructModel)
params={'epochs':[20]}
tune=GridSearchCV(estimator=kr,param_grid=params,cv=5,refit=True)
tuned=tune.fit(tr_features,tr_labels)

print (tuned.best_score_)
print (tuned.best_params_)

# evaluate model with standardized dataset
res=tune.predict(ts_features)

res=scale_labels.inverse_transform(res)
ts_labels=scale_labels.inverse_transform(ts_labels)

x=mean_absolute_error(ts_labels, res)

print ('MAE::NN',x)
