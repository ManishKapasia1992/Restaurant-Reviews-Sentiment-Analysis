import pandas as pd
import numpy as np
import pickle

# Loading the dataset
df = pd.read_csv(r'C:\Users\admin\Desktop\Restaurant_Reviews.txt', delimiter='\t', quoting=3)
# print(df.head())
# print(df.shape)
print(df.columns)

import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from nltk.stem import PorterStemmer

corpus = []
ps = PorterStemmer()

# Cleaning the reviews

for i in range(0, df.shape[0]):
    # Remove the unwanted texts
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df.Review[i])

    review = review.lower()

    # Convert into tokens
    review_words = review.split()

    # Remove the stop words
    review_words = [word for word in review_words if word not in set(stopwords.words('english'))]

    # Stemming of the words
    review = [ps.stem(word) for word in review_words]

    # convert to review
    review = ' '.join(review)

    corpus.append(review)

# print(corpus)


# Create a bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df['Liked'].values

# Create a pickle file for Countvectorizer

filename1 = r'C:\Users\admin\Desktop\Restaurant_Reviews_Sentmental_Anlysis_cv_transform.pkl'
pickle.dump(cv, open(filename1, 'wb'))

# Model building

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB(alpha=0.2)
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

# Creating a pickle file for model
filename2 = r'C:\Users\admin\Desktop\Restaurant_Reviews_Sentmental_Anlysis_nbclassifier_model.pkl'
pickle.dump(nb_classifier, open(filename2, 'wb'))

from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))