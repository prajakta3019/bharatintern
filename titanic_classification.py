#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import pearsonr


# In[5]:


data = pd.read_csv(r"C:\Users\Dell\Desktop\Titanic-Dataset.csv")


# In[6]:


data


# In[10]:


data.shape


# In[11]:


data.info()


# In[12]:


data.isnull().sum()


# In[13]:


# Dropping the "Cabin" column from the dataframe
data = data.drop(columns='Cabin', axis=1)
data.head()


# In[64]:


data.dtypes


# In[66]:


data.duplicated().sum()


# In[68]:


data.isnull().sum().sort_values(ascending=False)*100/len(data)


# In[ ]:





# In[14]:


# Replacing the missing vlaues in "Age" column with mean 
data['Age'].fillna(data['Age'].mean(), inplace=True)


# In[15]:


# Finding the mode value of "Embarkked" Column 
print(data['Embarked'].mode())


# In[16]:


print(data['Embarked'].mode()[0]) # 0 is the index


# In[17]:


# Replacing the missing values in "Embarked" column with the mode value
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


# In[19]:


data.isnull().sum()


# In[20]:


# Getting statistical measures about the data (It's not useful while handling categorical column)
data.describe()


# In[21]:


data['Survived'].value_counts()


# In[78]:


data.describe(include='O')


# In[79]:


data = data['Sex'].value_counts()
data


# In[86]:


# Plotting Percantage Distribution of Sex Column
plt.figure(figsize=(5,5))
plt.pie(data.values,labels=data.index,autopct='%.2f%%')
plt.legend()
plt.show()


# In[22]:


sns.set()


# In[24]:


# Count plot for Survived Column
sns.countplot(x='Survived', data=data)


# In[72]:


# Age Distribution
sns.kdeplot(x=data['Age'])
plt.show()


# In[73]:


# Showinf Distribution of Age Survived Wise
sns.kdeplot(x=data['Age'],hue=data['Survived'])
plt.show()


# In[25]:


data['Sex'].value_counts()


# In[29]:


sns.countplot(x='Sex', data=data)


# In[30]:


# Number of survivors by Gender wise
sns.countplot(x="Sex", hue="Survived", data=data)


# In[31]:


# Creating a Count plot for Pclass Column 
sns.countplot(x="Pclass", data=data)


# In[32]:


# Number of Survivers by Pclass wise 
sns.countplot(x="Pclass", hue="Survived", data=data)


# In[74]:


# Plotting Histplot for Dataset
data.hist(figsize=(10,10))
plt.show()


# In[76]:


# Plotting Boxplot for dataset
# Checking for outliers
sns.boxplot(data)
plt.show()


# In[34]:


data["Embarked"].value_counts()


# In[35]:


# Converting Categorical Columns 
data.replace({'Sex':{'male':0, 'female':1}, 'Embarked':{'S':0, 'C':1, 'Q':2}}, inplace=True)


# In[36]:


data.head()


# In[37]:


X = data.drop(columns=["PassengerId", "Name", "Ticket", "Survived"], axis=1)
Y = data['Survived']


# In[38]:


print(X)


# In[39]:


print(Y)


# In[40]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[41]:


print(X.shape, X_train.shape, X_test.shape)


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[43]:


model = LogisticRegression()
model.fit(X_train, Y_train)


# In[44]:


model_prediction = model.predict(X_test)


# In[45]:


accuracy_score(model_prediction, Y_test)


# In[46]:


confusion_matrix(Y_test, model_prediction)


# In[47]:


results = pd.DataFrame({
    'Model': ['Logistic Regression'], 
    'Score': [0.78]
})

results


# In[48]:


model_prediction


# In[49]:


#To save the model in a pkl file. 

import pickle as pkl

pkl.dump(model, open('model.pkl', 'wb'))


# In[50]:


print(X)


# In[51]:


print(Y)


# In[52]:


X_train.iloc[0,:]


# In[53]:


a = list(X_train.iloc[0,:])
a = np.array(a)


# In[54]:


ypred = model.predict(a.reshape(-1, 7))
ypred 


# In[55]:


Y_train[0]


# In[56]:


loaded_model = pkl.load(open('model.pkl', 'rb'))


# In[57]:


type(loaded_model)


# In[58]:


ypred = loaded_model.predict(a.reshape(-1,7))
ypred


# In[101]:


import pickle
import streamlit as st  
import numpy as np 

model_file = pickle.load(openb(bharatintern\'model.pkl','rb'))

def pred_output(user_input): 
    model_input = np.array(user_input)
    ypred = model_file.predict(model_input.reshape(-1,7))
    return ypred[0]


def main(): 
    st.title("Titanic Classification - rubangino.in")

    # Input Variables 
    passenger_class = st.text_input("Enter the passenger class: (1/2/3)")

    sex = st.text_input("Enter your sex (Male/Female): ")
    if sex == "Male" or sex == "male": 
        sex = 0
    elif sex == "Female" or sex == "female": 
        sex = 1
    else: 
        st.error('Invalid Input!', icon="🚨")

    # st.success(sex)

    age = st.text_input("Enter their age: ")

    sibsp = st.text_input("Enter their Siblings: ")

    parch = st.text_input("Enter their parch: ")

    fare = st.text_input("Enter their ticket Fare: ")

    embarked = st.text_input("Enter their Port of Embarked: (C=Cherbourg | Q=Queentown | S=Southampton) ")
    if embarked == "C" or embarked == "c": 
        embarked = 1
    elif embarked == "S" or embarked == "s": 
        embarked = 0
    elif embarked == "Q" or embarked == "q": 
        embarked = 2
    else: 
        st.error("Invalid Input!", icon="🚨")

    # Button to predict
    if st.button('Predict'): 
        user_input = [passenger_class, sex, age, sibsp, parch, fare, embarked]
        make_prediction = pred_output(user_input)  

        if make_prediction == 0: 
            make_prediction = "Not Survived :("
        elif make_prediction == 1: 
            make_prediction = "Survived :)"

        st.success(make_prediction)

if __name__ == '__main__': 
    main()


# SEX - Male=0 Female=1
# Embarked - C=Cherbourg Q=Queentown S=Southampton


# In[62]:


get_ipython().system('pip install streamlit')


# In[ ]:




