# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
source=pd.read_csv('../input/zillow-prize-1/train_2016_v2.csv',low_memory=False)
source=source.sort_values('parcelid')
sam_sub=pd.read_csv('../input/zillow-prize-1/sample_submission.csv')
sam_sub=sam_sub.rename(columns={'ParcelId':'parcelid'})
sam_sub=sam_sub.sort_values('parcelid')

windowStartOct=['2016-07-01','2016-08-01','2016-09-01']
windowEndOct=['2016-10-01','2016-10-01','2016-10-01']

windowStartNov=['2016-08-01','2016-09-01','2016-10-01']
windowEndNov=['2016-11-01','2016-11-01','2016-11-01']

windowStartDec=['2016-09-01','2016-10-01','2016-11-01']
windowEndDec=['2016-12-01','2016-12-01','2016-12-01']

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
            return dfc
        
        def Window():
                variableWindow=source[(source.transactiondate >= windowStart[i]) & (source.transactiondate< windowEnd[i])] 
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
        for k in range(3):              
            if (int(windowEnd[i].split('-')[1])-int(windowStart[i].split('-')[1])==3):
                    ThreeMonthWindow=retVarThree()
                    res1=pd.merge(ThreeMonthWindow,sam_sub,on='parcelid').drop(['201610','201611','201612','201710','201711','201712'],axis=1)
                    print (res1)
    
            if (int(windowEnd[i].split('-')[1])-int(windowStart[i].split('-')[1])==2):
                    TwoMonthWindow=retVarTwo()
                    res2=pd.merge(TwoMonthWindow,sam_sub,on='parcelid').drop(['201610','201611','201612','201710','201711','201712'],axis=1)
                    print (res2)
    
            if (int(windowEnd[i].split('-')[1])-int(windowStart[i].split('-')[1])==1):
                    OneMonthWindow=retVarOne()
                    res3=pd.merge(OneMonthWindow,sam_sub,on='parcelid').drop(['201610','201611','201612','201710','201711','201712'],axis=1)
                    print (res3)
                    pd.DataFrame(pd.concat([res1,res2.drop(['parcelid'],axis=1,inplace=True),res3.drop(['parcelid'],axis=1,inplace=True)],axis=1)).to_csv('EngineeredFeatures'+str(k)+'.csv',index=False,header=['parcelid','mean','median','std','min','max','quantile'])
             
         
slidingWindow(windowStartOct,windowEndOct) 
slidingWindow(windowStartNov,windowEndNov) 
slidingWindow(windowStartDec,windowEndDec) 


#----------extracting static features------------------

props=pd.read_csv('../input/zillow-prize-1/properties_2016.csv')
props=props.sort_values('parcelid')

static_features=props.loc[sam_sub.index]

static_features.to_csv('ExtractedPredictionStaticFeatures.csv',index=False)

#-------merging data--------------
windowFeatures=['EngineeredFeatures0.csv','EngineeredFeatures1.csv','EngineeredFeatures2.csv']
for l in range(3):
    pdf=pd.read_csv(windowFeatures[l])
    predDataset=pd.merge(static_features,pdf,on='parcelid')
    predDataset.to_csv('PredDataset'+str(l)+'.csv',index=False)
    
    
#----------Handling categorical values----------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

replacements=['PredDataset0.csv','PredDataset1.csv','PredDataset2.csv']
maps=['../input/dataset0/Dataset0.csv','../input/dataset1/Dataset1.csv','../input/dataset2/Dataset2.csv']
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
        
        dff=pd.read_csv(filepath_or_buffer=maps[i])
        dff=dff.loc[:,['parcelid','logerror']]
        dff=dff.loc[df1.index]
        df1=pd.concat([df1,dff.drop('parcelid',axis=1,inplace=True)],axis=1)
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
               u'censustractandblock', u'mean',u'median',u'std',u'min',u'max',u'quantile']]
        
        print (df10.shape)
        
        df10.to_csv('FinalPredictionDataset'+str(i)+'.csv',index=False)
            
HandleCatValues()

dataset=['FinalPredictionDataset0.csv','FinalPredictionDataset1.csv','FinalPredictionDataset2.csv']

for data in range(len(dataset)): 
    df111=pd.read_csv(filepath_or_buffer=dataset[data])
    df111=df111.fillna(404)
    df111.to_csv('FinalPredictionDataset'+str(data)+'.csv',index=False)


#-------------------------Making predictions using XGB-------------------------
    from sklearn.preprocessing import Imputer
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error
    import joblib
    import numpy as np
    TestSet=np.loadtxt('FinalPredictionDataset'+str(data)+'.csv',delimiter=',',skiprows=1)
    
    ts_features = TestSet[:,1:64]
    #imputer=Imputer(missing_values=404,strategy='median')
    #ts_features=imputer.fit_transform(ts_features)

    test=xgb.DMatrix(data=ts_features)
    
    loaded_model = joblib.load("../input/trainmodel/xgboost.joblib.dat")
    
    preds=loaded_model.predict(test)
    
    df=pd.DataFrame(np.round(preds,4),columns=['logerror'],index=None)
    
    df.to_csv('PredictionsXGB'+str(data)+'.csv',index=False)
'''
#----------------------------------Making predictions using RF----------------------
TestSet=loadtxt('testSetAfterAddingClusterMeans-OctPred.csv',delimiter=',',skiprows=1)

ts_features = TestSet[:,1:64]
test=xgb.DMatrix(data=ts_features)

loaded_model = joblib.load("xgboost.joblib.dat")

preds=loaded_model.predict(test)

df=pd.DataFrame(numpy.round(preds,4),columns=['logerror'],index=None)

df.to_csv('Predictions(XGBoost)(AfterOutlierRemoval)-Oct16.csv',index=False)
'''
#-----------------------------sample submission --------------------------------

df1=pd.read_csv('PredictionsXGB0.csv')
df2=pd.read_csv('PredictionsXGB1.csv')
df3=pd.read_csv('PredictionsXGB2.csv')
df7=pd.read_csv('../input/zillow-prize-1/sample_submission.csv')
df6=df7['ParcelId']
df7=df7['201610']
df4 = pd.concat([df6,df1, df2,df3,df7,df7,df7], axis=1,ignore_index=True)
df4=pd.DataFrame(df4)
df4.to_csv('Submission.csv',index=False,header=['ParcelId','201610','201611','201612','201710','201711','201712'])

#---------------------------Rounding off----------------------------------------------

df=pd.read_csv('Submission.csv')

df1=df.round(4)

df1.to_csv('Submission.csv',index=False)
