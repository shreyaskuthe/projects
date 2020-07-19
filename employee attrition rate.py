import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#%%
df=pd.read_csv(r'C:\Users\user\Desktop\hack\Dataset\Train.csv',index_col=0)
print(df.isnull().sum())
#%%
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
#%%
print(df.head())
#%%
print(df.shape)
#%%
df.info()
#%%%
print(df.describe(include="all"))
#%%
for col in df.columns: 
    if sum(df[col].isnull())/float(len(df.index)) >=0.3: del df[col]
#%%
print(df.shape)
#%%
print(df.shape)
#%%
print(df.dtypes)
#%%
for x in df.columns[:]:
        if df[x].dtype=='object':
           df[x].fillna(df[x].mode()[0],inplace=True)
        elif df[x].dtype=='int64' or df[x].dtype=='float64':
            df[x].fillna(df[x].mean(),inplace=True)
#%%
print(df.isnull().sum())
#%%
colname=[]
for x in df.columns:
    if df[x].dtype=='object':
        colname.append(x)
colname
#%%label encoding object variable in train data 
from  sklearn import preprocessing
le=preprocessing.LabelEncoder()
for x in colname:
        df[x]=le.fit_transform(df[x])   
#%%
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:] 
#%%
sns.distplot(Y,hist=True)    
#%%
Y_log=np.sqrt(Y)
#%%
sns.distplot(Y_log,hist=True)
#%%
from sklearn import preprocessing
# separate the data from the target attributes
# normalize the data attributes
X = preprocessing.normalize(X)
#%%
X = preprocessing.scale(X)
#%%
print(df)
#%%
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]
#%%
from sklearn.model_selection import train_test_split
#Split the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
#%%
from sklearn.tree import DecisionTreeRegressor
#%%
regressor=DecisionTreeRegressor(criterion='friedman_mse',splitter='best',
                             min_samples_split=2,
                                min_samples_leaf=30,min_weight_fraction_leaf=0.0, 
                                max_features='sqrt', random_state=1234, max_leaf_nodes=5, 
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                presort='deprecated')
#%%
regressor.fit(X_train,Y_train)
#%%
Y_pred=regressor.predict(X_test)
print(Y_pred)
#%%
new_df=pd.DataFrame()
new_df=X_test
new_df['Actual']=Y_test
new_df['Predicted']=Y_pred
print(new_df)
#%%
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
r2=r2_score(Y_test,Y_pred)
print(r2)
rmse=np.sqrt(mean_squared_error(Y_test,Y_pred))
print(rmse)
adjusted_r_squared = 1 - (1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print(adjusted_r_squared)
#%%%
print(min(Y_test))
print(max(Y_test))
#%%
test=pd.read_csv(r'C:\Users\user\Desktop\hack\Dataset\test.csv',index_col=0)
#%%
print(test.isnull().sum())
#%%
for x in test.columns[:]:
        if test[x].dtype=='object':
           test[x].fillna(df[x].mode()[0],inplace=True)
        elif test[x].dtype=='int64' or test[x].dtype=='float64':
            test[x].fillna(test[x].mean(),inplace=True)
#%%
print(test.isnull().sum())
#%%
colname=[]
for x in test.columns:
    if test[x].dtype=='object':
        colname.append(x)
colname
#%%label encoding object variable in train data 
from  sklearn import preprocessing
le=preprocessing.LabelEncoder()
for x in colname:
        test[x]=le.fit_transform(test[x])  
#%%
pred=regressor.predict(test)
print(pred)
#%%
pred1=pd.DataFrame(data=pred)
#%%
pred1.to_csv('pred32.csv')