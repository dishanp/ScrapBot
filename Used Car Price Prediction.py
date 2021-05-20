#!/usr/bin/env python
# coding: utf-8

# # Used Car Price Prediction🏎️
# ### By Dishan Purkayastha

# Importing Required Packages

# In[2]:


import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# Importing dataset

# In[3]:


data=pd.read_csv('cars.csv')


# Basic Analysis Of Dataset:

# In[4]:


data.head(3)


# In[5]:


data.shape


# In[6]:


data.columns


# In[7]:


data.info


# ##### Dataset Analysis On Various Parameters:

# In[8]:


loc = data.groupby('Location')[['Unnamed: 0']].count()
loc =loc.sort_values('Unnamed: 0', ascending=False).reset_index()
loc.rename(columns = {'Unnamed: 0':'Cars'},inplace=True)
loc

labels = list(loc.Location)
plt.figure(figsize=(10,10))
plt.title("Location Wise Distribution",fontweight='bold',fontsize=30)
plt.tick_params(labelsize=40)
plt.pie(loc.Cars,labels=labels,textprops={'fontsize': 13});
plt.savefig('loaction.png', dpi=300)


# In[9]:


loc = data.groupby('Fuel_Type')[['Unnamed: 0']].count()
loc =loc.sort_values('Unnamed: 0', ascending=False).reset_index()
loc.rename(columns = {'Unnamed: 0':'Cars'},inplace=True)
loc

labels = list(loc.Fuel_Type)
plt.figure(figsize=(5,5))
plt.title("Fuel Type Wise Distribution",fontweight='bold',fontsize=30)
plt.tick_params(labelsize=40)
plt.pie(loc.Cars,labels=labels,textprops={'fontsize': 13});
plt.savefig('fueltype.png', dpi=300)


# In[10]:


loc = data.groupby('Year')[['Unnamed: 0']].count()
loc =loc.sort_values('Unnamed: 0', ascending=False).reset_index()
loc.rename(columns = {'Unnamed: 0':'Cars'},inplace=True)
loc


# In[11]:


plt.figure(figsize=(20,10))
plt.xlabel('Year')
plt.ylabel('Cars')
plt.title('Year wise distribution');
plt.bar(loc.Year,loc.Cars, color='#bf88be');
plt.savefig('yearwise.png', dpi=300)


# In[12]:


data.columns


# In[13]:


data.Transmission.unique()


# In[44]:


loc = data.groupby('Transmission')[['Unnamed: 0']].count()
loc =loc.sort_values('Unnamed: 0', ascending=False).reset_index()
loc.rename(columns = {'Unnamed: 0':'Cars'},inplace=True)
loc

labels = list(loc.Transmission)
plt.figure(figsize=(5,5))
plt.title("Transimssion Type Wise Distribution",fontweight='bold',fontsize=30)
plt.tick_params(labelsize=40)
plt.pie(loc.Cars,labels=labels,textprops={'fontsize': 13});
plt.savefig('manual.png', dpi=300)


# In[43]:


loc = data.groupby('Owner_Type')[['Unnamed: 0']].count()
loc =loc.sort_values('Unnamed: 0', ascending=False).reset_index()
loc.rename(columns = {'Unnamed: 0':'Cars'},inplace=True)
loc

labels = list(loc.Owner_Type)
plt.figure(figsize=(5,5))
plt.title("Owner Type Wise Distribution",fontweight='bold',fontsize=30)
plt.tick_params(labelsize=40)
plt.pie(loc.Cars,labels=labels,textprops={'fontsize': 13});
plt.savefig('trans.png', dpi=300)


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], 
                                                    data.iloc[:, -1], 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# In[18]:


X_train.shape


# ### Refining Dataset

# In[19]:


#Removing the first colum,as it is not of any use
X_train = X_train.iloc[:, 1:]
X_test = X_test.iloc[:, 1:]


# In[20]:


X_train['Name'].value_counts()


# #### Deriving Manufacturer name from each car name,since it is an important paramter to determine re-sale value
# 

# In[21]:


make_train = X_train["Name"].str.split(" ", expand = True)
make_test = X_test["Name"].str.split(" ", expand = True)
X_train["Manufacturer"] = make_train[0]
X_test["Manufacturer"] = make_test[0]


# In[54]:


X_train.head()


# In[22]:


plt.figure(figsize = (20, 10))
plot = sns.countplot(x = 'Manufacturer', data = X_train)
plt.xticks(rotation = 90)
for p in plot.patches:
    plot.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                        ha = 'center', 
                        va = 'center', 
                        xytext = (0, 5),
                        textcoords = 'offset points')

plt.title("Distribution based on manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("Count of cars")


# In[23]:


#Dropping Name & location as parameters as they are not determinants of price
X_train.drop("Name", axis = 1, inplace = True)
X_test.drop("Name", axis = 1, inplace = True)
X_train.drop("Location", axis = 1, inplace = True)
X_test.drop("Location", axis = 1, inplace = True)


# In[24]:


#Deriving & Adding Car Age(In Years) To The Datasets
curr_time = datetime.datetime.now()
X_train['Year'] = X_train['Year'].apply(lambda x : curr_time.year - x)
X_test['Year'] = X_test['Year'].apply(lambda x : curr_time.year - x)


# In[25]:


#Extracting numeric value of mileage
mileage_train = X_train["Mileage"].str.split(" ", expand = True)
mileage_test = X_test["Mileage"].str.split(" ", expand = True)

X_train["Mileage"] = pd.to_numeric(mileage_train[0], errors = 'coerce')
X_test["Mileage"] = pd.to_numeric(mileage_test[0], errors = 'coerce')


# In[26]:


#Let's check for missing values.

print(sum(X_train["Mileage"].isnull()))
print(sum(X_test["Mileage"].isnull()))


# In[27]:


#Replacing Missing Values With Mean Value
X_train["Mileage"].fillna(X_train["Mileage"].astype("float64").mean(), inplace = True)
X_test["Mileage"].fillna(X_train["Mileage"].astype("float64").mean(), inplace = True)


# In[28]:


#Extracting numeric value of engine capacity & power
cc_train = X_train["Engine"].str.split(" ", expand = True)
cc_test = X_test["Engine"].str.split(" ", expand = True)

bhp_train = X_train["Power"].str.split(" ", expand = True)
bhp_test = X_test["Power"].str.split(" ", expand = True)


# In[29]:


X_train["Engine"] = pd.to_numeric(cc_train[0], errors = 'coerce')
X_test["Engine"] = pd.to_numeric(cc_test[0], errors = 'coerce')

X_train["Power"] = pd.to_numeric(bhp_train[0], errors = 'coerce')
X_test["Power"] = pd.to_numeric(bhp_test[0], errors = 'coerce')


# In[30]:


#Replacing All Missing Values With Mean Value

X_train["Engine"].fillna(X_train["Engine"].astype("float64").mean(), inplace = True)
X_test["Engine"].fillna(X_train["Engine"].astype("float64").mean(), inplace = True)

X_train["Power"].fillna(X_train["Power"].astype("float64").mean(), inplace = True)
X_test["Power"].fillna(X_train["Power"].astype("float64").mean(), inplace = True)

X_train["Seats"].fillna(X_train["Seats"].astype("float64").mean(), inplace = True)
X_test["Seats"].fillna(X_train["Seats"].astype("float64").mean(), inplace = True)


# In[31]:


#Dropping New_price as a parameter since most values are missing from the dataset
X_train.drop(["New_Price"], axis = 1, inplace = True)
X_test.drop(["New_Price"], axis = 1, inplace = True)


# Data Processing

# In[32]:


#Creating Dummy Columns for categorical columns before training.
X_train = pd.get_dummies(X_train,
                         columns = ["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],
                         drop_first = True)

X_test = pd.get_dummies(X_test,
                         columns = ["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],
                         drop_first = True)


# In[34]:


#Replacing Missing Columns With Zeroes
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]


# In[35]:


#Scaling The Values Before Training
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)


# ### Training and Prediction:

# Linear Regression Model:

# In[36]:


linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)
y_pred = linearRegression.predict(X_test)
r2_score(y_test, y_pred)


# Random Forest Model

# In[42]:


rf = RandomForestRegressor(n_estimators = 100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
r2_score(y_test, y_pred)


# ##### Clearly,the Random Forest model performed better with an R2 score of 0.88
