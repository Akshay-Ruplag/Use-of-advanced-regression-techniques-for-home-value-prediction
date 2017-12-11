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

TrainSet=np.loadtxt('../input/Trainset(12).csv',delimiter=',',skiprows=1)
print(TrainSet.shape)
features = TrainSet[:,1:99]
labels = TrainSet[:,99]

tr_features, ts_features, tr_labels, ts_labels = train_test_split(features,labels, test_size=0.30, random_state=42)

imputer=Imputer(missing_values=-191,strategy='median')
tr_features=imputer.fit_transform(tr_features)
ts_features=imputer.transform(ts_features)

tr_labels=tr_labels.reshape(-1,1)
ts_labels=ts_labels.reshape(-1,1)

scale_features=StandardScaler()
scale_labels=StandardScaler()

tr_features=scale_features.fit_transform(tr_features)
ts_features=scale_features.transform(ts_features)

tr_labels=scale_labels.fit_transform(tr_labels)
ts_labels=scale_labels.transform(ts_labels)

# create model
def constructModel(lrs=0.1,hl1=100,hl2=50,do=0.4):
    model = Sequential()
    model.add(Dense(98, input_dim=98, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(do))
    model.add(Dense(hl1,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(do))
    model.add(Dense(hl2,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(do))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    # Compile model
    adam=Adam(lr=lrs)
    model.compile(loss='mae',optimizer=adam ,metrics=['mae'])
    return model


#if (df.isnull())

params={'lrs':[0.1],'epochs':[100],'hl1':[100],'hl2':[50],'do':[0.4]}
kr=KerasRegressor(build_fn=constructModel)
tune=GridSearchCV(estimator=kr,param_grid=params,cv=5,refit=True)
tuned=tune.fit(tr_features,tr_labels)

print (tuned.best_score_)
print (tuned.best_params_)

# evaluate model with standardized dataset
res=tune.predict(ts_features)

res=scale_labels.inverse_transform(res)
ts_labels=scale_labels.inverse_transform(ts_labels)

x=mean_absolute_error(ts_labels, res)

print ('MAE',x)
