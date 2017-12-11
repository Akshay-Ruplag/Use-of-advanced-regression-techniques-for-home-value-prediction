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
#---------------------------XGB model(Base Learner1)-------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
import joblib
import xgboost as xgb

#train=pd.read_csv('')
TrainSet=np.loadtxt('../input/trainset10/Trainset(10).csv',delimiter=',',skiprows=1)
print(TrainSet.shape)
features = TrainSet[:,1:91]
labels = TrainSet[:,91]

tr_features, ts_features, tr_labels, ts_labels = train_test_split(features,labels, test_size=0.30, random_state=42)

imputer=Imputer(missing_values=-191,strategy='median')
tr_features=imputer.fit_transform(tr_features)
ts_features=imputer.transform(ts_features)

data=xgb.DMatrix(data=tr_features,label=tr_labels)
test=xgb.DMatrix(data=ts_features)

eval_set=xgb.DMatrix(ts_features,ts_labels)

params={'booster':'gbtree','eta':0.1,'seed':0,'subsample':0.8,'colsample_bytree':0.8,'objective':'reg:linear','max_depth':3,'min_child_weight':2,'silent':1,'eval_metric':'mae','verbose':True,'gamma':0}
bst=xgb.train(params,data,num_boost_round=7000,evals=[(eval_set,'eval')],early_stopping_rounds=10)
res=xgb.cv(params = params, dtrain = data, num_boost_round = 7000, nfold = 10, metrics = ['mae'],early_stopping_rounds = 10)

y_pred=bst.predict(test)

y_pred=pd.DataFrame(y_pred)


#--------------------------catboost Regressor(Base Learner2)---------------------------------------
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from catboost import Pool, CatBoostRegressor
import catboost as cb

imputer=Imputer(missing_values=-191,strategy='most_frequent')
tr_features=imputer.fit_transform(tr_features)
ts_features=imputer.transform(ts_features)
pool=Pool(data=tr_features,label=tr_labels)
print (ts_features.shape)

# specify the training parameters
model = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.01, loss_function='MAE',eval_metric='MAE',od_type="Iter",od_wait=10)
#train the model
eval_set=(ts_features,ts_labels)
model.fit(tr_features,tr_labels,use_best_model=True,eval_set=eval_set)
#model.get_feature_importance(tr_features,tr_labels)
#plt.show()
params={'iterations':600,'depth':6,'learning_rate':0.01,'eval_metric':'MAE'}
res=cb.cv(params=params,pool=pool,fold_count=5,inverted=False,partition_random_seed=0,shuffle=True)
# make the prediction using the resulting model
preds = model.predict(ts_features)

preds=pd.DataFrame(preds)
actual=pd.DataFrame(ts_labels)

finaldataset=pd.concat([y_pred,preds,actual],axis=1)
finaldataset.to_csv('finaldataset.csv',index=False)

#-----------------------------Final Learner----------------------------------
from sklearn.tree import DecisionTreeRegressor

source=np.loadtxt('finaldataset.csv',delimiter=',',skiprows=1)

features = source[:,0:2]
labels = source[:,2]

tr_features, ts_features, tr_labels, ts_labels = train_test_split(features,labels, test_size=0.30, random_state=42)

stacker = DecisionTreeRegressor(criterion='mae',max_depth=3)
stacker.fit(tr_features,tr_labels)
predictions=stacker.predict(ts_features)

print (mean_absolute_error(ts_labels,predictions))
