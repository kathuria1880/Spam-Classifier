#!/usr/bin/env python
# coding: utf-8

# In[8]:


#importing the dataset
import pandas as pd
messages=pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',names=['label','message'])


# In[9]:


#printing the dataset
messages


# In[10]:


#finding the size of dataset
messages.shape


# In[11]:


#Data Cleaning and Preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]


# In[12]:


for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)


# In[13]:


#Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()


# In[16]:


y=pd.get_dummies(messages['label'])#converting values in form of 0 and 1
y=y.iloc[:,1].values


# In[17]:


y


# In[18]:


#Trsain Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)


# In[20]:


#Training model using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,y_train)
y_pred=spam_detect_model.predict(X_test)


# In[21]:


y_pred


# In[22]:


#Making Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)


# In[23]:


confusion_m


# In[24]:


#finding the accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)


# In[25]:


accuracy


# ### We got the Accuracy of 97%.
# ### In Case if we do not get high accuracy we can apply TF-IDF model in place of Bag Of Words.

# In[ ]:




