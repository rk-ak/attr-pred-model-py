
# coding: utf-8

# In[176]:

# Attrition Prediction Model
# Import required modules
import pandas as pd
import numpy as np


# In[177]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[178]:

# import input dataset
employee_df = pd.read_csv('C:\\Users\\xxxxxx\\Attrition Prediction Model - Input - Sample.csv')


# In[179]:

employee_df


# In[180]:

employee_df.head(5)


# In[181]:

employee_df.tail(5)


# In[182]:

employee_df.info()


# In[183]:

employee_df.describe()


# In[184]:

# drop unwanted fields
employee_df = employee_df.drop("Attrition_Probability", axis=1)
employee_df = employee_df.drop("Bank_ID", axis=1)
employee_df = employee_df.drop("Period_Key", axis=1)
employee_df = employee_df.drop("business_function", axis=1)


# In[185]:

# encode required description fields
employee_df['Actual_Attrition'] = employee_df['Actual_Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['is_consistent_good_performer'] = employee_df['is_consistent_good_performer'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['OT hours'] = employee_df['OT hours'].apply(lambda x: 1 if x == 'Yes' else 0)

attrition_df = employee_df[employee_df['Actual_Attrition'] == 1]
active_df = employee_df[employee_df['Actual_Attrition'] == 0]

employee_df.describe()


# In[186]:

# check overall stats from dataset
print("Total = ", len(employee_df))

print("Attrition = ", len(left_df))
print("Attrition % = ", 1.*len(left_df)/len(employee_df)*100.0, "%")
 
print("Active = ", len(stayed_df))
print("Active % = ", 1.*len(stayed_df)/len(employee_df)*100.0, "%")


# In[187]:

# segregate description fields
desc_df = employee_df[['region', 'global_grade','frontline_nonfrontline']]
desc_df


# In[188]:

# encode description fields
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
desc_df = onehotencoder.fit_transform(desc_df).toarray()


# In[189]:

desc_df.shape


# In[190]:

desc_df = pd.DataFrame(desc_df)


# In[191]:

desc_df 


# In[192]:

# segregate numeric fields
num_df = employee_df[['yrs_since_last_promotion', 'is_consistent_good_performer', 'is_high_performer','service_tenure', 'age', 'gender', 'marital_status', 'highest_education_level', 'Number of years under current manager']]
num_df


# In[193]:

# merge numeric and description fields(after encoding)
employee_all = pd.concat([num_df, desc_df], axis = 1)
employee_all


# In[194]:

# scale numeric fields
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(employee_all)


# In[195]:

# define independent variables dataset : X => all encoded fields
X


# In[196]:

# define dependent variable field : y => Actual Attrition field
y = employee_df['Actual_Attrition']
y


# In[197]:

# split datasets for training and testing with 25% ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[198]:

X_train.shape


# In[199]:

X_test.shape


# In[200]:

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=250)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[201]:

y_pred


# In[202]:

# check model accuracy
from sklearn.metrics import confusion_matrix, classification_report
print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))


# In[203]:

# check confusion matrix
cm = confusion_matrix(y_pred, y_test)
cm


# In[204]:

# check precision and recall
print(classification_report(y_test, y_pred))


# In[205]:

#Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=250)
model.fit(X_train, y_train)


# In[206]:

y_pred = model.predict(X_test)


# In[207]:

# check accuracy
from sklearn.metrics import confusion_matrix, classification_report
print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))


# In[208]:

# check confusion matrix
cm = confusion_matrix(y_pred, y_test)
cm


# In[209]:

# check precision and recall
print(classification_report(y_test, y_pred))


# In[ ]:



