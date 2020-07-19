#%%importing base libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#%%import data file
loan1=pd.read_csv(r'C:\Users\hp\Desktop\capston project\Python Project - Bank Lending\XYZCorp_LendingData.txt',delimiter= '\t')

#%%
#%%checking heads
pd.set_option('display.max_columns',None)
loan1.head()
#%% display missing values
pd.set_option('display.max_rows',None)
loan1.isnull().sum()
#%%checking shape of the data
loan1.shape
#%% getting info of the data
loan1.info()
#%% to get description of data data
loan1.describe(include="all")
#%% filling missing values in column 'verification_status_joint' with null
for value in ['verification_status_joint']:
    loan1[value].fillna(loan1[value].fillna('null'),inplace=True)
pd.set_option('display.max_rows',None)
loan1.isnull().sum()
#%% filling missing values in 'annual_inc_joint','dti_joint' & 'display.max_rows'
for value in ['annual_inc_joint','dti_joint']:
    loan1[value].fillna(loan1[value].fillna(0),inplace=True)
pd.set_option('display.max_rows',None)
loan1.isnull().sum()
#%%% droppingg columns which are having more than 30% missing values
for col in loan1.columns: 
    if sum(loan1[col].isnull())/float(len(loan1.index)) >=0.3: del loan1[col]
#%% cross checking missing values
loan1.isnull().sum()
#%% checking shape of the data after deleting columns which are having more than 30% missing values
loan1.shape
#%% feature selection based on domain knowledge
loan2=loan1.drop(['addr_state','collection_recovery_fee','collections_12_mths_ex_med','last_pymnt_d','inq_last_6mths','recoveries','earliest_cr_line','emp_title','id','initial_list_status','last_credit_pull_d','member_id','next_pymnt_d','policy_code','purpose','pymnt_plan','revol_bal','revol_util','sub_grade','title','zip_code'],axis=1)
#%%% checkingn description of data after feature selection
#%%% getting corelation  matrix
corr_df=loan2.corr()
corr_df
print(corr_df)
#%%#%%% checkingn description of data after feature selection
loan2.describe(include="all")
loan2.shape
#%% checking data types of all the variables
loan2.dtypes
#%% treating missing values
for x in loan2.columns[:]:
        if loan2[x].dtype=='object':
           loan2[x].fillna(loan2[x].mode()[0],inplace=True)
        elif loan2[x].dtype=='int64' or loan2[x].dtype=='float64':
            loan2[x].fillna(loan2[x].mean(),inplace=True)
#%% cross checking missing values
loan2.isnull().sum()
#%%  cross checking heads of the data
loan2.head()
#%% checking data types
loan2.dtypes
#%% data partition according to problm statement
loan2.issue_d=pd.to_datetime(loan2.issue_d)
col_name='issue_d'
print (loan2[col_name].dtype)
#%% splitting data into test and train using "issue_d" variable
split_date="May-2015"
loan2_train=loan2.loc[loan2['issue_d']<=split_date]
loan2_train=loan2_train.drop(['issue_d'],axis=1)
loan2_train.shape
#%%
loan2_test=loan2.loc[loan2['issue_d']> split_date]
loan2_test=loan2_test.drop(['issue_d'],axis=1)
loan2_test.shape
#%%#% viewing columns which are having object data type for label encoding
colname=[]
for x in loan2.columns:
    if loan2[x].dtype=='object':
        colname.append(x)
colname
#%%label encoding object variable in train data 
from  sklearn import preprocessing
le=preprocessing.LabelEncoder()
for x in colname:
        loan2_train[x]=le.fit_transform(loan2_train[x])
#%% label encoding object variable in test data 
from  sklearn import preprocessing
le=preprocessing.LabelEncoder()
for x in colname:
        loan2_test[x]=le.fit_transform(loan2_test[x])
#%% printing heads to check that all the variable get label encoded in the train data
loan2_train.head()
#%%printing heads to check that all the variable get label encoded in the test data
loan2_train.head()
#%% creating X(independent) variable and Y (dedependent) variable for model building
X_train=loan2_train.values[:,:-1]
Y_train=loan2_train.values[:,-1]

print(Y_train)
X_test=loan2_test.values[:,:-1]
Y_test=loan2_test.values[:,-1]

print(Y_test)
#%% scaling train data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X=scaler.transform(X_train)
print(X_train)
#%% scaling test data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_test)
X=scaler.transform(X_test)
print(X_test)
#%% as  dependent variable having classes we are using logistic regression 
#%% importing logistic regression from sklearn library
from sklearn.linear_model  import LogisticRegression 
#%% Creating model 
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)
#%%
Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))  #getiing list of predictions in console
print(classifier.coef_) 
print(classifier.intercept_)
#%%
"""importing confusion_matrix- to check predictions on the basis of type 1 & 2 erros
accuracy_score- to check how accurately model predicting,
classification_report-to drill down complete model prediction """
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%% model tuning to reduce type 2 error
#%% tuning of model logistic regression /adjusting threshold
## store the predicted probabilities
Y_pred_prob=classifier.predict_proba(X_test)
print(Y_pred_prob)
#%%
Y_pred_class=[]
for value in Y_pred_prob[:,1]:
    if value > 0.75:
        Y_pred_class.append(1)
    else:
        Y_pred_class.append(0)
print(Y_pred_class)  
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred_class)
print(cfm)
print(classification_report(Y_test,Y_pred_class))
acc=accuracy_score(Y_test,Y_pred_class)
print('Accuracy of the model:',acc)
#%%
for a in np.arange(0.4,0.61,0.01):
    predict_mine = np.where(Y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :",
          cfm[1,0]," , type 1 error:", cfm[0,1]) 
#%%
#%% auc-0.5-0.6-poor model,0.6-0.7-badmodel,0.7-00.8-good model,0.8-0.9-vgood model,0.9-1.0-excellent model
from sklearn import metrics

fpr, tpr, z = metrics.roc_curve(Y_test, Y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc)
#%%
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
#%% cross validation  k fold
#Using cross validation
#model evaluation
classifier=LogisticRegression()

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=30,random_state=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%% 
#model tuning
for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])


Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred
                   )
print('Accuracy of the model:',acc)
#%%%%
#predicting using the bagging classifier
from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier(10,random_state=30)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%
#predicting using the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=101,random_state=10)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%############ as data is skewd try using mannual label encoding and log transform################
###################################################################################################
##################################################################################################
#%%
#importing libraries
import pandas as pd 
import numpy as np
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
#%%%
#importing dataframe
loan1=pd.read_csv(r'C:\Users\hp\Desktop\capston project\Python Project - Bank Lending\XYZCorp_LendingData.txt',delimiter= '\t')
#%%%
#for viewing all the columns
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
#%%%printi9ng heads
loan1.head()
print(loan1.columns)
#%%% getting shape of the data
loan1.shape
#%%% describe all variables
loan1.describe(include="all")
#%%
#Replacing the missing values in verification_status_joint with 'null'
for value in ['verification_status_joint']:
    loan1[value].fillna(loan1[value].fillna('null'),inplace=True)
pd.set_option('display.max_rows',None)
loan1.isnull().sum()
#%%
##Replacing the missing values in annual_inc_joint and dti_joint with '0'
for value in ['annual_inc_joint','dti_joint']:
    loan1[value].fillna(loan1[value].fillna(0),inplace=True)
#%%
# specifying the dependent variable
loan1['default_ind'].value_counts
#%%%
#ploting the count graph of the dependent variable
sns.countplot('default_ind',data=loan1)
#%%%
#finding the correlation of the dependent variable with other variables
list=loan1[loan1.columns].corr()['default_ind'][:]
loan1[loan1.columns].corr()['default_ind'][:]

#%%%
#eliminanting the columns with very low correltion with the dependent variable
lis=[]
for i in range(len(lis)):
    if lis [i]<0.02 and lis [i]>(-0.02):
        lis.append(lis.index[i])
#%%%
for i in range (len(lis)):
    del loan1['{}'.format(lis[i])]
#%%%
#checking the correlation of the dependent variable with other variables   
loan1[loan1.columns].corr()['default_ind'][:]
#%%%shecking shape of the data
loan1.shape
#%%%
#checking the missing values
loan1.isnull().sum()
#%%%
#deleting the columns which have missing values more than 30%
for col in loan1.columns: 
    if sum(loan1[col].isnull())/float(len(loan1.index)) >=0.3: del loan1[col]
#%%% checking shape oh the data
loan1.shape
#%%% checking missing values
loan1.isnull().sum()
#%%% checjing info of the ddata
loan1.info()
#%%%
# manual lable encoding of the 'term' variable
loan1['term'].value_counts()
#%%%
sns.countplot('term', data=loan1, hue='default_ind')
#%%%
loan1['term']=np.where(loan1['term']==' 36 months', 0,loan1['term'])
loan1['term']=np.where(loan1['term']==' 60 months', 1,loan1['term'])
#%%%
loan1['term'].value_counts()
#%%%
loan1['term']=loan1['term'].astype(float)
#%%%
# manual lable encoding of the 'grade' variable
loan1['grade'].value_counts()
#%%%
sns.countplot('grade',data=loan1,hue='default_ind')
#%%%
loan1['grade']=np.where(loan1['grade']=='A',0,loan1['grade'])
loan1['grade']=np.where(loan1['grade']=='B',0,loan1['grade'])
loan1['grade']=np.where(loan1['grade']=='C',0,loan1['grade'])
loan1['grade']=np.where(loan1['grade']=='D',1,loan1['grade'])
loan1['grade']=np.where(loan1['grade']=='E',1,loan1['grade'])
loan1['grade']=np.where(loan1['grade']=='F',1,loan1['grade'])
loan1['grade']=np.where(loan1['grade']=='G',1,loan1['grade'])
#%%%
loan1['grade'].value_counts()
#%%%
loan1['grade']=loan1['grade'].astype(float)
#%%%
## manual lable encoding of the 'emp_length' variable
loan1['emp_length'].value_counts()
#%%% printing count plot
sns.countplot('emp_length',data=loan1,hue='default_ind')
#%%%
loan1['emp_length']=np.where(loan1['emp_length']=='10+ years',0,loan1['emp_length'])
loan1['emp_length']=np.where(loan1['emp_length']=='2 years',1,loan1['emp_length'])
loan1['emp_length']=np.where(loan1['emp_length']=='< 1 year',1,loan1['emp_length'])
loan1['emp_length']=np.where(loan1['emp_length']=='3 years',1,loan1['emp_length'])
loan1['emp_length']=np.where(loan1['emp_length']=='1 year',1,loan1['emp_length'])
loan1['emp_length']=np.where(loan1['emp_length']=='5 years',0,loan1['emp_length'])
loan1['emp_length']=np.where(loan1['emp_length']=='4 years',0,loan1['emp_length'])
loan1['emp_length']=np.where(loan1['emp_length']=='7 years',0,loan1['emp_length'])
loan1['emp_length']=np.where(loan1['emp_length']=='8 years',0,loan1['emp_length'])
loan1['emp_length']=np.where(loan1['emp_length']=='6 years',0,loan1['emp_length'])
loan1['emp_length']=np.where(loan1['emp_length']=='9 years',0,loan1['emp_length'])
#%%%
loan1['emp_length'].value_counts()
#%%%
loan1['emp_length']=loan1['emp_length'].astype(float)
#%%%
## manual lable encoding of the 'home_ownership' variable
loan1['home_ownership'].value_counts()
#%%% printing count plot
sns.countplot('home_ownership',data=loan1,hue='default_ind')
#%%%
loan1['home_ownership']=np.where(loan1['home_ownership']=='MORTGAGE',1,loan1['home_ownership'])
loan1['home_ownership']=np.where(loan1['home_ownership']=='RENT',1,loan1['home_ownership'])
loan1['home_ownership']=np.where(loan1['home_ownership']=='OWN',0,loan1['home_ownership'])
loan1['home_ownership']=np.where(loan1['home_ownership']=='OTHER',0,loan1['home_ownership'])
loan1['home_ownership']=np.where(loan1['home_ownership']=='NONE',0,loan1['home_ownership'])
loan1['home_ownership']=np.where(loan1['home_ownership']=='ANY',0,loan1['home_ownership'])
#%%%
loan1['home_ownership'].value_counts()
#%%%
loan1['home_ownership']=loan1['home_ownership'].astype(float)
#%%%
# manual lable encoding of the 'verification_status' variable
loan1['verification_status'].value_counts()
#%%% printing count plot
sns.countplot('verification_status', data=loan1,hue='default_ind')
#%%%
loan1['verification_status']=np.where(loan1['verification_status']=='Source Verified',0,loan1['verification_status'])
loan1['verification_status']=np.where(loan1['verification_status']=='Verified',0,loan1['verification_status'])
loan1['verification_status']=np.where(loan1['verification_status']=='Not Verified',1,loan1['verification_status'])
#%%%
loan1['verification_status'].value_counts()
#%%%
loan1['verification_status']=loan1['verification_status'].astype(float)
#%%% printing count plot
## manual lable encoding of the 'verification_status_joint' variable
loan1['verification_status_joint'].value_counts()
#%%%
sns.countplot('verification_status_joint',data=loan1,hue='default_ind')
#%%%
loan1['verification_status_joint']=np.where(loan1['verification_status_joint']=='null',0,loan1['verification_status_joint'])
loan1['verification_status_joint']=np.where(loan1['verification_status_joint']=='Verified',0,loan1['verification_status_joint'])
loan1['verification_status_joint']=np.where(loan1['verification_status_joint']=='Source Verified',0,loan1['verification_status_joint'])
loan1['verification_status_joint']=np.where(loan1['verification_status_joint']=='Not Verified',1,loan1['verification_status_joint'])
#%%%
loan1['verification_status_joint'].value_counts()
#%%%
loan1['verification_status_joint']=loan1['verification_status_joint'].astype(float)
#%%% 
### manual lable encoding of the 'application_type' variable
loan1['application_type'].value_counts()
#%%% printing count plot
sns.countplot('application_type', data=loan1,hue='default_ind')
#%%%
loan1['application_type']=np.where(loan1['application_type']=='INDIVIDUAL',0,loan1['application_type'])
loan1['application_type']=np.where(loan1['application_type']=='JOINT',1,loan1['application_type'])
#%%%
loan1['application_type'].value_counts()
#%%%
loan1['application_type']=loan1['application_type'].astype(float)
#%%%
loan1.info()
#%%%
#feature selection based on domain knowledge
loan2=loan1.drop(['addr_state','collection_recovery_fee','collections_12_mths_ex_med','last_pymnt_d','inq_last_6mths','recoveries','earliest_cr_line','emp_title','id','initial_list_status','last_credit_pull_d','member_id','next_pymnt_d','policy_code','purpose','pymnt_plan','revol_bal','revol_util','sub_grade','title','zip_code'],axis=1)
#%%% checking shape if the data
loan2.shape
#%%% checkung data types
loan2.dtypes
#%%% checking missing values
loan2.isnull().sum()
#%%%
#treating missing values in the categorical variable by reaplacing with mode
loan2['emp_length'].fillna(loan2['emp_length'].mode()[0],inplace=True)
#%%% checking missing values after treatment
loan2.isnull().sum()
#%%%
#treating missing values in the numerical variable by reaplacing with mean
loan2['tot_coll_amt'].fillna(loan2['tot_coll_amt'].mean(),inplace=True)
loan2['tot_cur_bal'].fillna(loan2['tot_cur_bal'].mean(),inplace=True)
loan2['total_rev_hi_lim'].fillna(loan2['total_rev_hi_lim'].mean(),inplace=True)
#%%%
loan2.isnull().sum()
#%%%
loan2.describe(include="all")
#%%%
#log transform
np.log(loan2['loan_amnt'])
#%%%
np.log(loan2['funded_amnt'])
#%%%
np.log(loan2['funded_amnt_inv'])
#%%%
np.log(loan2['int_rate'])
#%%%
np.log(loan2['installment'])
#%%%
np.log(loan2['total_pymnt'])
#%%%
np.log(loan2['total_pymnt_inv'])
#%%%
np.log(loan2['total_rec_prncp'])
#%%%
np.log(loan2['last_pymnt_amnt'])
#%%%
np.log(loan2['annual_inc_joint'])
#%%%
np.log(loan2['tot_cur_bal'])
#%%%
np.log(loan2['total_rev_hi_lim'])
#%% data partion
#splitting the data in train and test
#data is splitted on the basis of issue_d i.e. May-2015
loan2.issue_d=pd.to_datetime(loan2.issue_d)
col_name='issue_d'
print (loan2[col_name].dtype)
#%%
split_date="May-2015"
loan2_train=loan2.loc[loan2['issue_d']<=split_date]
loan2_train=loan2_train.drop(['issue_d'],axis=1)
loan2_train.shape
#%%
loan2_test=loan2.loc[loan2['issue_d']> split_date]
loan2_test=loan2_test.drop(['issue_d'],axis=1)
#%%%
#model building
X_train=loan2_train.values[:,:-1]
Y_train=loan2_train.values[:,-1]

print(Y_train)
X_test=loan2_test.values[:,:-1]
Y_test=loan2_test.values[:,-1]
#%%%
#calling the algorithm
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)
print(classifier.coef_)
print(classifier.intercept_)
#%%% checking accuracy of the data
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%
# tuning of model logistic regression /adjusting threshold
# store the predixted probabilities
Y_pred_prob=classifier.predict_proba(X_test)
print(Y_pred_prob)
#%%%
Y_pred_class=[]
for value in Y_pred_prob[:,1]:
    if value > 0.75:
        Y_pred_class.append(1)
    else:
            Y_pred_class.append(0)
print(Y_pred_class)
#%%% checking accuracy after threshold adjustment
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred_class)
print(cfm)
print(classification_report(Y_test,Y_pred_class))
acc=accuracy_score(Y_test,Y_pred_class)
print('Accuracy of the model:',acc)
#%%% 
#to decide threshold
for a in np.arange(0.4,0.61,0.01):
    predict_mine = np.where(Y_pred_prob[:,1] > a, 1, 0)
cfm=confusion_matrix(Y_test, predict_mine)
total_err=cfm[0,1]+cfm[1,0]
print("Errors at threshold ", a, ":",total_err, " , type 2 error :",
cfm[1,0]," , type 1 error:", cfm[0,1])
#%%% 
#auc-0.5-0.6-poor model,0.6-0.7-badmodel,0.7-00.8-good model,0.8-0.9-vgood model,0.9-1.0-excellent model
from sklearn import metrics

fpr, tpr, z = metrics.roc_curve(Y_test, Y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc)
#%%% ROC curve

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
#%%%
# cross validation k fold
#Using cross validation
#model evaluation
classifier=LogisticRegression()

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=20,random_state=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%%%
#model tuning
for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])


Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%trying  decision tree classifier
from sklearn.tree import DecisionTreeClassifier
#%%%
#create the model object

model_DT=DecisionTreeClassifier(criterion='gini', max_depth=10,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=5, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=10, splitter='best')

#fit the model on the data and predict the values
model_DT.fit(X_train,Y_train)
#%%%
#predit using the model
Y_pred=model_DT.predict(X_test)
#print (Y_pred)
#print(list(zip(Y_test,Y_pred)))
#%%% checking accuracy of model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%% trying SVM
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=50,gamma=0.1)
svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)
print(list(Y_pred))
#%%%%
#predicting using the bagging classifier
from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier(100,random_state=30)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#%%% checking accuracy of model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%
#predicting using the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=101,random_state=10)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#%%%checking accuracy of model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%% using Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
model=AdaBoostClassifier(n_estimators=40,base_estimator=DecisionTreeClassifier(),random_state=10)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#%%% checking accuracy of model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%% using gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
model=GradientBoostingClassifier(n_estimators=250,random_state=10)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#%%% checking accuracy of model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%% Voating classifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

# create the sub models
estimators = []
#model1 = LogisticRegression()
#estimators.append(('log', model1))
model2 = DecisionTreeClassifier(criterion='gini',random_state=10)
estimators.append(('cart', model2))
model3 = SVC(kernel="rbf", C=50,gamma=0.1)
estimators.append(('svm', model3))
#model4 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
#estimators.append(('knn', model4))

#%%
# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#print(Y_pred)
#%%% checking accuracy of model
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%

#%%%
#1--names
#2-problem statment
#3-data cleaning-feature (selected nd deleted) nd missing
#4-correlatin
#5-y variable
#6-count plots
#7-skewnes
#8-lable encoding nd converion in date time
#8- data scalingss






