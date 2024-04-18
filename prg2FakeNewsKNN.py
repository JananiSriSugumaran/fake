#!/usr/bin/env python
#python 3.7 32 bit 
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#get_ipython().run_line_magic('matplotlib', 'inline')

#Import Cancer data from the Sklearn library
# Dataset can also be found here (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)

#from sklearn.datasets import load_breast_cancer
#cancer = load_breast_cancer()
#from google.colab import files
#uploaded = files.upload()
data = pd.read_csv("newssvm.csv")
def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


data.head()
cancer=data

data['title']=data['title'].str.replace("€", "")
data['title']=data['title'].str.replace("™", "")
data['title']=data['title'].str.replace("â", "")
data['text']=data['text'].str.replace("€", "")
data['text']=data['text'].str.replace("™", "")
data['text']=data['text'].str.replace("â", "")

data['text']=data['text'].str.replace(",", "")
data['title']=data['title'].str.replace(",", "")


data['category']=data['category'].astype('category')
data['title']=data['title'].astype('category')
data['text']=data['text'].astype('category')
data.replace(np.nan,0)
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
data = data.reset_index()
#newsline=strip_non_ascii(i)
# In[2]:


#print(cancer)
#print(cancer.columns)


# As we can see above, not much can be done in the current form of the dataset. We need to view the data in a better format.

# # Let's view the data in a dataframe.

# In[3]:


df = data#pd.DataFrame(np.c_[cancer['title'], cancer['category']], columns = np.append(cancer['category'], ['category']))

df.head()




df.shape


# As we can see,we have 596 rows (Instances) and 31 columns(Features)

# In[5]:


print(df.columns)
#df=df.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')
#df['category']=df['category'].astype('category')
#df['title']=df['title'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
cat_columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
print(df.head())
#exit()

# Above is the name of each columns in our dataframe.

# # The next step is to Visualize our data

# In[6]:


print(df.columns.values)
# Let's plot out just the first 5 variables (features)
#sns.pairplot(df, hue = 'category', vars = ['title',  'category'])


# **Note:** 
#     
#   1.0 (Orange) = Benign (No Cancer)
#   
#   0.0 (Blue) = Malignant (Cancer)

# # How many Benign and Malignant do we have in our dataset?

# In[8]:


df['category'].value_counts()


# As we can see, we have 212 - Malignant, and 357 - Benign

#  Let's visulaize our counts

# In[9]:


sns.countplot(df['category'], label = "Count") 


# # Let's check the correlation between our features 

# In[10]:


plt.figure(figsize=(20,12)) 
sns.heatmap(df.corr(), annot=True) 




X = df.drop(['category'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model.
X.head()


# In[12]:


y = df['category']
y.head()


# # Create the training and testing data

# Now that we've assigned values to our "X" and "y", the next step is to import the python library that will help us to split our dataset into training and testing data.

# - Training data = Is the subset of our data used to train our model.
# - Testing data =  Is the subset of our data that the model hasn't seen before. This is used to test the performance of our model.

# In[13]:


from sklearn.model_selection import train_test_split


# Let's split our data using 80% for training and the remaining 20% for testing.

# In[14]:

indices =range(len(df))
leng=len(df)
X_train, X_test, y_train, y_test,tr,te = train_test_split(X, y,indices, test_size = 0.2, random_state = 20)


# Let now check the size our training and testing data.

# In[15]:


print ('The size of our training "X" (input features) is', X_train.shape)
print ('\n')
print ('The size of our testing "X" (input features) is', X_test.shape)
print ('\n')
print ('The size of our training "y" (output feature) is', y_train.shape)
print ('\n')
print ('The size of our testing "y" (output features) is', y_test.shape)


# # Import Support Vector Machine (SVM) Model 

# In[16]:


from sklearn.svm import SVC


# In[17]:


svc_model = SVC()


# # Now, let's train our SVM model with our "training" dataset.

# In[18]:


svc_model.fit(X_train, y_train)


# # Let's use our trained model to make a prediction using our testing data

# In[19]:


y_predict = svc_model.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


# In[21]:


cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_fake', 'is_real'],
                         columns=['predicted_fake','predicted_real'])
print(confusion)


# In[22]:


sns.heatmap(confusion, annot=True)


# In[23]:


print(classification_report(y_test, y_predict))




X_train_min = X_train.min()
X_train_min


# In[25]:


X_train_max = X_train.max()
X_train_max


# In[26]:


X_train_range = (X_train_max- X_train_min)
X_train_range


# In[27]:


X_train_scaled = (X_train - X_train_min)/(X_train_range)
X_train_scaled.head()


# # Normalize Training Data

# In[28]:


X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range


# In[29]:


svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[30]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)


# # SVM with Normalized data

# In[31]:


cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_fake', 'is_real'],
                         columns=['predicted_fake','predicted_real'])
confusion


# In[32]:


sns.heatmap(confusion,annot=True,fmt="d")


# In[33]:
print(classification_report(y_test,y_predict))
for i in range(0,len(y_predict)):
   print(te[i] , ":",  y_predict[i])
ac = accuracy_score(y_test,y_predict)
print('Accuracy:')
print(round(ac,2))
print('Confusion Matrix')
print(confusion_matrix(y_test, y_predict))
exit()
