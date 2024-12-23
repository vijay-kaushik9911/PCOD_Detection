import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score

dataset=pd.read_csv('PCOS.csv')
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])
le=LabelEncoder()
X[:,3]=le.fit_transform(X[:,3])

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[4])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(X_train,y_train,batch_size=32,epochs=100)

print(ann.predict(sc.transform([[0.0, 1.0,0.0, 27,60, 1,1, 3.2, 10.5,15.8]]))>0.5)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)
