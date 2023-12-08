#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection & Processing

# In[2]:


# load the data from csv file to Pandas DataFrame
titanic_data = pd.read_csv('train.csv')


# In[3]:


# printing the first 5 rows of the dataframe
titanic_data.head()


# In[4]:


# number of rows and Columns
titanic_data.shape


# In[5]:


# getting some informations about the data
titanic_data.info()


# In[6]:


# check the number of missing values in each column
titanic_data.isnull().sum()


# Handling the Missing values

# In[7]:


# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)


# In[8]:


# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[9]:


# finding the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())


# In[10]:


print(titanic_data['Embarked'].mode()[0])


# In[11]:


# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[12]:


# check the number of missing values in each column
titanic_data.isnull().sum()


# Data Analysis

# In[13]:


# getting some statistical measures about the data
titanic_data.describe()


# In[14]:


# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()


# Data Visualization

# In[15]:


sns.set()


# In[16]:


# making a count plot for "Survived" column

sns.countplot(x='Survived', data=titanic_data)
plt.show()


# In[17]:


titanic_data['Sex'].value_counts()


# In[18]:


# making a count plot for "Sex" column
sns.countplot(x='Sex', data=titanic_data)


# In[19]:


# number of survivors Gender wise
sns.countplot(x='Sex', hue='Survived', data=titanic_data)


# In[20]:


# making a count plot for "Pclass" column
sns.countplot(x='Pclass', data=titanic_data)


# In[21]:


sns.countplot(x='Pclass', hue='Survived', data=titanic_data)


# Encoding the Categorical Columns

# In[22]:


titanic_data['Sex'].value_counts()


# In[23]:


titanic_data['Embarked'].value_counts()


# In[24]:


# converting categorical Columns

titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[25]:


titanic_data.head()


# Separating features & Target

# In[26]:


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']


# In[27]:


print(X)


# In[28]:


print(Y)


# Splitting the data into training data & Test data

# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[30]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# Logistic Regression

# In[31]:


model = LogisticRegression()


# In[32]:


# training the Logistic Regression model with training data
model = LogisticRegression(max_iter=1000) 
model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[33]:


# accuracy on training data
X_train_prediction = model.predict(X_train)


# In[34]:


print(X_train_prediction)


# In[35]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[36]:


# accuracy on test data
X_test_prediction = model.predict(X_test)


# In[37]:


print(X_test_prediction)


# In[38]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:




