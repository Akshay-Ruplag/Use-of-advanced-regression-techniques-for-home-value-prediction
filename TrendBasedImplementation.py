# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
from sklearn.preprocessing import OneHotEncoder 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df_properties16=pd.read_csv('../input/zillow-prize-1/properties_2016.csv',low_memory=False)
df_train16=pd.read_csv('../input/zillow-prize-1/train_2016_v2.csv',parse_dates=['transactiondate'],low_memory=False)
print(df_train16.describe())
df_train16=df_train16[(df_train16.transactiondate >= '2016-01-01') & (df_train16.transactiondate<= '2016-09-30')]
df_train16=df_train16[(df_train16.logerror> -2) & (df_train16.logerror<2)]
df_samplesub=pd.read_csv('../input/zillow-prize-1/sample_submission.csv',low_memory=False)

df_train16['Month']=df_train16['transactiondate'].dt.month
df_train16['Day']=df_train16['transactiondate'].dt.day
df_train16['TimeComponent']=df_train16['Month']/df_train16['Day']



df_properties16['NullCount']=df_properties16.isnull().sum(axis=1)

#life of property
df_properties16['N-life'] = 2016 - df_properties16['yearbuilt']

#error in calculation of the finished living area of home
df_properties16['N-LivingAreaError'] = df_properties16['calculatedfinishedsquarefeet']/df_properties16['finishedsquarefeet12']

#proportion of living area
df_properties16['N-LivingAreaProp'] = df_properties16['calculatedfinishedsquarefeet']/df_properties16['lotsizesquarefeet']
df_properties16['N-LivingAreaProp2'] = df_properties16['finishedsquarefeet12']/df_properties16['finishedsquarefeet15']

#Amout of extra space
df_properties16['N-ExtraSpace'] = df_properties16['lotsizesquarefeet'] - df_properties16['calculatedfinishedsquarefeet'] 
df_properties16['N-ExtraSpace-2'] = df_properties16['finishedsquarefeet15'] - df_properties16['finishedsquarefeet12'] 

#Total number of rooms
df_properties16['N-TotalRooms'] = df_properties16['bathroomcnt']*df_properties16['bedroomcnt']


# Number of Extra rooms
df_properties16['N-ExtraRooms'] = df_properties16['roomcnt'] - df_properties16['N-TotalRooms'] 

#Ratio of the built structure value to land area
df_properties16['N-ValueProp'] = df_properties16['structuretaxvaluedollarcnt']/df_properties16['landtaxvaluedollarcnt']

#Does property have a garage, pool or hot tub and AC?
df_properties16['N-GarPoolAC'] = ((df_properties16['garagecarcnt']>0) & (df_properties16['pooltypeid10']>0) & (df_properties16['airconditioningtypeid']!=5))*1 

df_properties16["N-location"] = df_properties16["latitude"] + df_properties16["longitude"]
df_properties16["N-location-2"] = df_properties16["latitude"]*df_properties16["longitude"]
df_properties16["N-location-2round"] = df_properties16["N-location-2"].round(-4)

df_properties16["N-latitude-round"] = df_properties16["latitude"].round(-4)
df_properties16["N-longitude-round"] = df_properties16["longitude"].round(-4)

#Ratio of tax of property over parcel
df_properties16['N-ValueRatio'] = df_properties16['taxvaluedollarcnt']/df_properties16['taxamount']

#TotalTaxScore
df_properties16['N-TaxScore'] = df_properties16['taxvaluedollarcnt']*df_properties16['taxamount']

#polnomials of tax delinquency year
df_properties16["N-taxdelinquencyyear-2"] = df_properties16["taxdelinquencyyear"] ** 2
df_properties16["N-taxdelinquencyyear-3"] = df_properties16["taxdelinquencyyear"] ** 3

#Length of time since unpaid taxes
df_properties16['N-life'] = 2016 - df_properties16['taxdelinquencyyear']

#Ratio of tax of property over parcel
df_properties16['N-ValueRatio'] = df_properties16['taxvaluedollarcnt']/df_properties16['taxamount']

#TotalTaxScore
df_properties16['N-TaxScore'] = df_properties16['taxvaluedollarcnt']*df_properties16['taxamount']

#polnomials of tax delinquency year
df_properties16["N-taxdelinquencyyear-2"] = df_properties16["taxdelinquencyyear"] ** 2
df_properties16["N-taxdelinquencyyear-3"] = df_properties16["taxdelinquencyyear"] ** 3

#Length of time since unpaid taxes
df_properties16['N-life'] = 2016 - df_properties16['taxdelinquencyyear']

#Number of properties in the zip
zip_count = df_properties16['regionidzip'].value_counts().to_dict()
df_properties16['N-zip_count'] = df_properties16['regionidzip'].map(zip_count)

#Number of properties in the city
city_count = df_properties16['regionidcity'].value_counts().to_dict()
df_properties16['N-city_count'] = df_properties16['regionidcity'].map(city_count)

#Number of properties in the city
region_count = df_properties16['regionidcounty'].value_counts().to_dict()
df_properties16['N-county_count'] = df_properties16['regionidcounty'].map(region_count)



#polnomials of the variable
df_properties16["N-structuretaxvaluedollarcnt-2"] = df_properties16["structuretaxvaluedollarcnt"] ** 2
df_properties16["N-structuretaxvaluedollarcnt-3"] = df_properties16["structuretaxvaluedollarcnt"] ** 3

#Average structuretaxvaluedollarcnt by city
group = df_properties16.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
df_properties16['N-Avg-structuretaxvaluedollarcnt'] = df_properties16['regionidcity'].map(group)

#Deviation away from average
df_properties16['N-Dev-structuretaxvaluedollarcnt'] = abs((df_properties16['structuretaxvaluedollarcnt'] - df_properties16['N-Avg-structuretaxvaluedollarcnt']))/df_properties16['N-Avg-structuretaxvaluedollarcnt']

df_properties16["location"] = df_properties16["latitude"] + df_properties16["longitude"]

df_properties16["AreaRatio1"]=df_properties16['finishedsquarefeet12']/df_properties16['lotsizesquarefeet']

df_properties16["AreaRatio2"]=df_properties16['garagetotalsqft']/df_properties16['lotsizesquarefeet']

df_properties16['AreaRatio3']=df_properties16['poolsizesum']/df_properties16['lotsizesquarefeet']

df_properties16['AreaRatio4']=df_properties16['landtaxvaluedollarcnt']/df_properties16['lotsizesquarefeet']

df_properties16['AreaRatio5']=df_properties16['taxamount']/df_properties16['lotsizesquarefeet']



ext=pd.DataFrame(df_properties16,columns=['parcelid','fireplaceflag','propertycountylandusecode','propertyzoningdesc','hashottuborspa','taxdelinquencyflag'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ext['fireplaceflag']=le.fit_transform(ext['fireplaceflag'].astype('str'))
ext['hashottuborspa']=le.fit_transform(ext['hashottuborspa'].astype('str'))
ext['propertycountylandusecode']=le.fit_transform(ext['propertycountylandusecode'].astype('str'))
ext['propertyzoningdesc']=le.fit_transform(ext['propertyzoningdesc'].astype('str'))
ext['taxdelinquencyflag']=le.fit_transform(ext['taxdelinquencyflag'].astype('str'))
#print("classes",le.classes_)
#catfeatures.to_csv('catFeatures.csv',index=False,header=None)
#catfeatures=pd.read_csv('catFeatures.csv',header=None)
#Encoder=OneHotEncoder()
#ins=Encoder.fit_transform(catfeatures)
#print(ins.toarray())

df_properties16=df_properties16.drop(['fireplaceflag','propertycountylandusecode','propertyzoningdesc','hashottuborspa','taxdelinquencyflag'],axis=1)
df_properties16=pd.DataFrame(pd.merge(df_properties16,ext,on='parcelid'))



import gc
gc.collect()

dataset=pd.merge(df_properties16,df_train16,on='parcelid')
print(dataset.head(5))
dataset=dataset.drop(['transactiondate'],axis=1)
dataset=dataset.fillna(-191)
print(dataset.shape)
dataset=dataset[['parcelid', 'airconditioningtypeid', 'architecturalstyletypeid',
       'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid',
       'buildingqualitytypeid', 'calculatedbathnbr', 'decktypeid',
       'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
       'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
       'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
       'fullbathcnt', 'garagecarcnt', 'garagetotalsqft',
       'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet',
       'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
       'propertylandusetypeid', 'rawcensustractandblock', 'regionidcity',
       'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt',
       'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid',
       'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt',
       'numberofstories', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
       'assessmentyear', 'landtaxvaluedollarcnt', 'taxamount',
       'taxdelinquencyyear', 'censustractandblock','propertycountylandusecode','propertyzoningdesc', 'NullCount', 'N-life',
       'N-LivingAreaError', 'N-LivingAreaProp', 'N-LivingAreaProp2',
       'N-ExtraSpace', 'N-ExtraSpace-2', 'N-TotalRooms', 'N-ExtraRooms',
       'N-ValueProp', 'N-GarPoolAC', 'N-location', 'N-location-2',
       'N-location-2round', 'N-latitude-round', 'N-longitude-round',
       'N-ValueRatio', 'N-TaxScore', 'N-taxdelinquencyyear-2',
       'N-taxdelinquencyyear-3', 'N-zip_count', 'N-city_count',
       'N-county_count', 'N-structuretaxvaluedollarcnt-2',
       'N-structuretaxvaluedollarcnt-3', 'N-Avg-structuretaxvaluedollarcnt',
       'N-Dev-structuretaxvaluedollarcnt', 'location', 'fireplaceflag',
       'hashottuborspa', 'taxdelinquencyflag',  'Month', 'Day',
       'TimeComponent','AreaRatio1','AreaRatio2','AreaRatio3','AreaRatio4','AreaRatio5','logerror']]

keys=dataset.keys()
for k in range(len(keys)):
    if(dataset[keys[k]].nunique()==1 or np.round((dataset[keys[k]].isnull().sum()/dataset.shape[0]),2)>0.90):
        dataset=dataset.drop(keys[k],axis=1)
    
dataset.to_csv('Trainset.csv',index=False)

from sklearn.cluster import KMeans 

frame=pd.read_csv('Trainset.csv',header=0)
cat1=frame['propertycountylandusecode']
cat2=frame['propertyzoningdesc']

def Send(cat):
    KM=KMeans(n_clusters=5)
    KM.fit(X=cat.values.reshape(-1,1))
    
    enc=pd.get_dummies(KM.labels_)
    print(type(enc))
    return enc 
    
cat1=Send(cat1)
cat2=Send(cat2)
frame=frame.drop(['propertycountylandusecode','propertyzoningdesc'],axis=1)
cat1=cat1.rename(index=int,columns={0:'a',1:'b',2:'c',3:'d',4:'e'})
cat2=cat2.rename(index=int,columns={0:'g',1:'h',2:'i',3:'j',4:'k'})
frame=pd.concat([frame,cat1,cat2],axis=1)


frame=frame[['parcelid', 'airconditioningtypeid', 'architecturalstyletypeid',
       'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid',
       'buildingqualitytypeid', 'calculatedbathnbr', 'decktypeid',
       'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
       'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
       'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
       'fullbathcnt', 'garagecarcnt', 'garagetotalsqft',
       'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet',
       'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
       'propertylandusetypeid', 'rawcensustractandblock', 'regionidcity',
       'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt',
       'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid',
       'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt',
       'numberofstories', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
        'landtaxvaluedollarcnt', 'taxamount',
       'taxdelinquencyyear', 'censustractandblock','a','b','c','d','e','g','h','i','j','k','NullCount', 'N-life',
       'N-LivingAreaError', 'N-LivingAreaProp',
       'N-ExtraSpace', 'N-TotalRooms', 'N-ExtraRooms',
       'N-ValueProp', 'N-GarPoolAC', 'N-location', 'N-location-2',
       'N-location-2round', 'N-latitude-round', 'N-longitude-round',
       'N-ValueRatio', 'N-TaxScore', 'N-taxdelinquencyyear-2',
       'N-taxdelinquencyyear-3', 'N-zip_count', 'N-city_count',
       'N-county_count', 'N-structuretaxvaluedollarcnt-2',
       'N-structuretaxvaluedollarcnt-3', 'N-Avg-structuretaxvaluedollarcnt',
       'N-Dev-structuretaxvaluedollarcnt', 'location', 'fireplaceflag',
       'hashottuborspa', 'taxdelinquencyflag',  'Month', 'Day',
       'TimeComponent','AreaRatio1','AreaRatio2','AreaRatio3','AreaRatio4','AreaRatio5','logerror']]
frame=frame.fillna(-191)

#---------------------------XGB model-------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
import joblib
import xgboost as xgb

#train=pd.read_csv('')
TrainSet=np.loadtxt('Trainset.csv',delimiter=',',skiprows=1)
print(TrainSet.shape)
features = TrainSet[:,1:99]
labels = TrainSet[:,99]

tr_features, ts_features, tr_labels, ts_labels = train_test_split(features,labels, test_size=0.30, random_state=42)

imputer=Imputer(missing_values=-191,strategy='median')
tr_features=imputer.fit_transform(tr_features)
ts_features=imputer.transform(ts_features)

data=xgb.DMatrix(data=tr_features,label=tr_labels)
test=xgb.DMatrix(data=ts_features)

eval_set=xgb.DMatrix(ts_features,ts_labels)

params={'booster':'gbtree','eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':3,'min_child_weight':2,'silent':1,'eval_metric':'mae','verbose':True,'gamma':0}
bst=xgb.train(params,data,num_boost_round=7000,evals=[(eval_set,'eval')],early_stopping_rounds=10)

y_pred=bst.predict(test)

res=xgb.cv(params = params, dtrain = data, num_boost_round = 7000, nfold = 10, metrics = ['mae'],early_stopping_rounds = 10)

print (res)

print (mean_absolute_error(ts_labels,y_pred))
joblib.dump(bst, "xgboost.joblib.dat")
#--------------------------catboost Regressor---------------------------------------
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt


TrainSet=np.loadtxt('Trainset.csv',delimiter=',',skiprows=1)

features = TrainSet[:,1:99]
labels = TrainSet[:,99]


tr_features, ts_features, tr_labels, ts_labels = train_test_split(features,labels, test_size=0.30, random_state=42)

from catboost import Pool, CatBoostRegressor
import catboost as cb

imputer=Imputer(missing_values=-191,strategy='most_frequent')
tr_features=imputer.fit_transform(tr_features)
ts_features=imputer.transform(ts_features)
pool=Pool(data=tr_features,label=tr_labels)
print (ts_features.shape)

# specify the training parameters
model = CatBoostRegressor(iterations=250, depth=6, learning_rate=0.01, loss_function='MAE',eval_metric='MAE',od_type="Iter",od_wait=10)
#train the model
eval_set=(ts_features,ts_labels)
model.fit(tr_features,tr_labels,use_best_model=True,eval_set=eval_set)
#model.get_feature_importance(tr_features,tr_labels)
#plt.show()
params={'iterations':250,'depth':6,'learning_rate':0.01,'eval_metric':'MAE'}
res=cb.cv(params=params,pool=pool,fold_count=5,inverted=False,partition_random_seed=0,shuffle=True)
print (res)
# make the prediction using the resulting model
preds = model.predict(ts_features)
#print(preds)
print (model.get_params())
print (mean_absolute_error(ts_labels,preds))
model.save_model("model")
print("shape",frame.shape)
frame.to_csv('Trainset.csv',index=False)
