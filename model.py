#!/usr/bin/env python
# coding: utf-8

# # Data Preparation
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import pickle
from nltk.stem.porter import PorterStemmer
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from imblearn.over_sampling import SMOTE

df = pd.read_csv('sample30.csv')

df.drop(columns=['reviews_userProvince','reviews_userCity'],inplace=True)
df.drop(columns=['reviews_date'],inplace=True)
df.user_sentiment.fillna(value='Positive',inplace=True)
df.reviews_didPurchase.fillna(value='NA',inplace=True)
df.reviews_doRecommend.fillna(value='NA',inplace=True)
df.reviews_username.fillna(value='DummyUser',inplace=True)
df.drop(columns=['manufacturer'],inplace=True)
df.reviews_title.fillna(value=' ',inplace=True)

# # Pre Proccessing

#Lets Create a new column by merging the review_text and title
df['review']= df['reviews_title']+ " "+ df['reviews_text']

#we can now drop the original columns as they are not needed now
df = df.drop(columns=['reviews_title','reviews_text'])
df['review'] = df['review'].str.lower()

#Removing Stopwords
def rem_stop(text):
  return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df['review']= df['review'].apply(lambda text: rem_stop(text))

PUNCT= string.punctuation
def rem_punct(text):
  return text.translate(str.maketrans('','',PUNCT))

df['review']= df['review'].apply(lambda text: rem_punct(text))

#Stemming


stemmer = PorterStemmer()
def stem(text):
  return " ".join([stemmer.stem(word) for word in text.split()])

df['review']= df['review'].apply(lambda text: stem(text))

df['user_sentiment']= df['user_sentiment'].apply(lambda x : 1 if x=='Positive' else 0)


vectorizer= TfidfVectorizer()
tfidf_model = vectorizer.fit_transform(df['review'].values)

X=df['review']
y=df['user_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#Transforming the X_train and X_test using the tf-idf model
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)



smt = SMOTE()
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)

log_model=LogisticRegression()

params={'C':np.logspace(-10, 1, 15),'class_weight':[None,'balanced'],'penalty':['l1','l2']}

cv = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)


# Create grid search using 5-fold cross validation
clf_LR = GridSearchCV(log_model, params, cv=cv, scoring='roc_auc', n_jobs=-1)

clf_LR.fit(X_train_sm, y_train_sm)

#Making pickle file of the model
pickle.dump(clf_LR,open('logreg.pkl','wb'))
pickle.dump(df,open('data.pkl','wb'))
pickle.dump(vectorizer,open('vectorizer.pkl','wb'))

# # Recommender System

# ## Test Train Split

# In[52]:


train, test = train_test_split(df, test_size=0.30, random_state=31)

dummy_train = train.copy()

#Products not rated by the user marked as 1 for prediction
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

#Creating Dummy Train Matrix
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(1)

# ##ITEM-ITEM Based Reccommendation
# 

# Taking the transpose of the rating matrix to normalize the rating around the mean for different products. In the user based similarity, we had taken mean for each user instead of each product.
df_pivot = train.pivot_table( 
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).T

### Normalising the rating of the movie for each user around 0 mean
mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T

#Creating the User Similarity Matrix using pairwise_distance function.
item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0

#only considering the positive corelation 
item_correlation[item_correlation<0]=0


item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)

#Using the dummy_train data to filter out the ratings
item_final_rating = np.multiply(item_predicted_ratings,dummy_train) 
#Making pickle file of the model
pickle.dump(item_final_rating,open('recommender.pkl','wb'))

