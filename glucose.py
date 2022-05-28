import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from xgboost import XGBClassifier
from xgboost import XGBClassifier
from xgboost import XGBClassifier
from sklearn import metrics 

df=pd.read_csv('../content/diabetes.csv')
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# model=XGBClassifier(gamma=0)
# model.fit(X_train,y_train)

# xg_pred=model.predict(X_test)

# print("Accuracy=",format(metrics.accuracy_score(y_test,xg_pred)))

model1 = Sequential()
model1.add(Dense(32,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(64,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(64,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(128,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(128,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(256,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(256,input_shape=(X_train.shape[1],)))
model1.add(Activation('relu'))
model1.add(Dense(2))
model1.add(Activation('softmax'))

model1.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",metrics=['accuracy'])
history=model1.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(X_test, y_test))
loss, accuracy = model1.evaluate(X_test,y_test, verbose=0)

def glucopredict(xt):
    score = model1.predict(xt)[0]
    return score  #rounding needed