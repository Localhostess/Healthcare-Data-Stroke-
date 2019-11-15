#!/usr/bin/env python
# coding: utf-8

# ## Importing Required Libraries

# In[1]:

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn import metrics


# ## Reading-in Datasets

# In[2]:

train_data = pd.read_csv('train_2v.csv')
test_data = pd.read_csv('test_2v.csv')

# In[3]:

test_data.head()

# In[4]:

train_data.head()

# In[5]:

print("Test Data Shape {}" .format(test_data.shape))

# In[6]:

print("Training Data Shape {}" .format(train_data.shape))

# In[7]:

train_data.describe()

# In[8]:


fig = plt.figure(figsize = (15,15))
ax = fig.gca()
train_data.hist(ax=ax)
plt.show()


# ## Data Preprocessing

# In[9]:


train_data.isnull().sum()/len(train_data)*100
#train_data BMI has over 3% of the data missing and over 30% of the smoking_status data missing


# In[10]:


test_data.isnull().sum()/len(train_data)*100
#train_data BMI has over 1% of the data missing and over 13% of the smoking_status data missing


# In[11]:


train_data["bmi"]= train_data["bmi"].fillna(train_data["bmi"].mean())
#fills cells with missing data with the average BMI of 28.6


# ## Handling Categorical Variables

# In[12]:


label = LabelEncoder()
train_data['gender'] = label.fit_transform(train_data['gender'])
train_data['ever_married'] = label.fit_transform(train_data['ever_married'])
train_data['work_type']= label.fit_transform(train_data['work_type'])
train_data['Residence_type']= label.fit_transform(train_data['Residence_type'])
#Label Encoder coverts categorical variables/features to 1s or 0s based on the variable value. Using train_data.shape()
#you can view this change#


# In[13]:


train_data.drop(["id"], axis = 1, inplace = True)
#Here I drop the id column from the dataset.


# In[14]:


#splitting data set into subjects with smoking status available or not. The goal is to build a seperate model for both
#and then work with the best one.
training_data_without_smokingstatus = train_data[train_data['smoking_status'].isnull()]


# In[15]:


training_data_with_smokingstatus = train_data[train_data['smoking_status'].notnull()]


# In[16]:


#drops the smoking status column for the null smoking status dataset
training_data_without_smokingstatus.drop(columns='smoking_status',axis=1,inplace=True)


# In[17]:


print("Training Data With Smoking Status Shape{}" .format(training_data_with_smokingstatus.shape))


# In[18]:


print("Training Data Without Smoking Status Shape{}" .format(training_data_without_smokingstatus.shape))


# In[19]:


training_data_with_smokingstatus['smoking_status'] = label.fit_transform(training_data_with_smokingstatus['smoking_status'])


# In[20]:


plt.figure(figsize=(12,7))
sns.heatmap(training_data_with_smokingstatus.corr('pearson'), annot = True)
#The Pearson correlation coefficient is the test statistics that measures the statistical relationship,
#or association, between two continuous variables.


# In[21]:


plt.figure(figsize=(12,7))
sns.heatmap(training_data_without_smokingstatus.corr('pearson'), annot = True)


# ## Handling Imbalanced Datasets

# In[22]:


training_data_with_smokingstatus['stroke'].value_counts()
#no stroke to stroke ratio is: 29470 to 638


# In[23]:


training_data_without_smokingstatus['stroke'].value_counts()
#no stroke to stroke ratio is: 13147 to 145


# In[24]:


#To handle the unbalanced dataset, the ROSE method is used.
ros = RandomOverSampler(random_state=0)


# In[25]:


X_resampled, y_resampled = ros.fit_sample(training_data_with_smokingstatus.loc[:,training_data_with_smokingstatus.columns!='stroke'], 
                                            training_data_with_smokingstatus['stroke'])                          


# In[26]:


print ('ROS Input Data Shape for Smoke Data: {}'.format(X_resampled.shape))
print ('ROS Output Data Shape for Smoke Data: {}'.format(y_resampled.shape))


# In[27]:


X_resampled1, y_resampled1 = ros.fit_sample(training_data_without_smokingstatus.loc[:,training_data_without_smokingstatus.columns!='stroke'], 
                                            training_data_without_smokingstatus['stroke'])                          


# In[28]:


print ('ROS Input Data Shape for Smoke Data: {}'.format(X_resampled1.shape))
print ('ROS Output Data Shape for Smoke Data: {}'.format(y_resampled1.shape))


# ## Train-Test-Split (With Smoking Status) 

# In[29]:


X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2)
print(X_train.shape)
print(X_test.shape)


# ### Logistic Regression Classifier

# In[30]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[31]:


predictions = logmodel.predict(X_test)


# In[32]:


print(classification_report(y_test,predictions))


# In[33]:


logmodel.score(X_test, y_test)


# In[34]:


log = LogisticRegression(penalty='l2', C=0.1)
log.fit(X_train,y_train)


# In[35]:


impFeatures = pd.DataFrame(log.coef_[0] ,index=training_data_with_smokingstatus.loc[:,training_data_with_smokingstatus.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
print (impFeatures)


# In[36]:


print(confusion_matrix(y_test,predictions))


# In[37]:


sns.heatmap(confusion_matrix(y_test,predictions), annot= True, cmap = 'viridis', fmt="2")
plt.title('Confusion Matrix')
plt.show()


# ### Gaussian NB Classifier

# In[38]:


gnb = GaussianNB()


# In[39]:


gnb.fit(X_train, y_train)


# In[40]:


y_prediction = gnb.predict(X_test)


# In[41]:


print("Accuracy:",metrics.accuracy_score(y_test, y_prediction))
#For the Dataset with Smoking Status it is observed that the Logistic Regression algorithm 
#performed better than the Gaussian NB.


# ## Train-Test-Split (Without Smoking Status)

# In[42]:


X_train1,X_test1,y_train1,y_test1 = train_test_split(X_resampled1,y_resampled1,test_size=0.2)
print(X_train1.shape)
print(X_test1.shape)


# ### Logistic Regression Classifier

# In[43]:


logmodel = LogisticRegression()
logmodel.fit(X_train1,y_train1)


# In[44]:


predictions1 = logmodel.predict(X_test1)


# In[45]:


print(classification_report(y_test1,predictions1))


# In[46]:


logmodel.score(X_test1, y_test1)


# In[47]:


log1 = LogisticRegression(penalty='l2', C=0.1)
log1.fit(X_train1,y_train1)


# In[48]:


impFeatures = pd.DataFrame(log1.coef_[0] ,index=training_data_without_smokingstatus.loc[:,training_data_without_smokingstatus.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
print(impFeatures)


# In[49]:


print(confusion_matrix(y_test1,predictions1))


# In[50]:


sns.heatmap(confusion_matrix(y_test1,predictions1), annot= True, cmap = 'viridis', fmt="2")
plt.title('Confusion Matrix')
plt.show()


# ### Gaussian NB Classifier

# In[51]:


gnb.fit(X_train1, y_train1)


# In[52]:


y_prediction1 = gnb.predict(X_test1)


# In[53]:


print("Accuracy:",metrics.accuracy_score(y_test1, y_prediction1))
#For the Dataset without Smoking Status it is observed that the Logistic Regression algorithm 
#performed better than the Gaussian NB.


#  ### Predicting The Target Variable (Stroke) Using The Logistic Model

# In[54]:


test_data["bmi"]=test_data["bmi"].fillna(test_data["bmi"].mean())


# In[55]:


test_data.drop(axis=1,columns=['smoking_status'],inplace=True)
#Smoking status dropped from the test data as it was noticed that the predictions for the dataset without smoking 
#status performed better.


# In[56]:


label = LabelEncoder()
test_data['gender'] = label.fit_transform(test_data['gender'])
test_data['ever_married'] = label.fit_transform(test_data['ever_married'])
test_data['work_type']= label.fit_transform(test_data['work_type'])
test_data['Residence_type']= label.fit_transform(test_data['Residence_type'])


# In[57]:


test_data.drop(["id"], axis = 1, inplace = True)


# In[58]:


pred = log1.predict(test_data)


# In[59]:


prediction = pd.DataFrame(pred,columns=['Pred'])


# In[60]:


prediction['Pred'].value_counts()


# In[ ]:




