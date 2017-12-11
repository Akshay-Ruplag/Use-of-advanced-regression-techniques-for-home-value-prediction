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

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import joblib
from sklearn.metrics import mean_absolute_error

TrainSet= np.loadtxt('../input/Trainset(12).csv', delimiter=',', skiprows=1)

features = TrainSet[:,1:99]
labels = TrainSet[:,99]

tr_features, ts_features, tr_labels, ts_labels = train_test_split(features,labels, test_size=0.30, random_state=42)

imputer=Imputer(missing_values=-191,strategy='median')
tr_features=imputer.fit_transform(tr_features)
ts_features=imputer.transform(ts_features)

#model creation and parameter configuration
model = RandomForestRegressor(n_estimators=30, criterion='mae', max_depth=3, min_samples_split=2, min_samples_leaf=50,  max_features='sqrt', oob_score=True, n_jobs=-1, random_state=1, verbose=0)

#training model on training data
model.fit(tr_features, tr_labels)

#prediction on test data
predicted_labels=model.predict(ts_features)

x=mean_absolute_error(ts_labels, predicted_labels)

print ("MeanAbsoluteError","%.4f" % x)

joblib.dump(model, "randomforest.joblib.dat")

cv=KFold(n=features.shape[0],n_folds=10,shuffle=True)
score=cross_val_score(model,features,labels,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)

print (np.mean(np.abs(score)))
