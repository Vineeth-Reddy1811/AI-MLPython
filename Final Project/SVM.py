import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer

stem = PorterStemmer()

#df = pd.read_csv('nlp.tsv', delimiter='\t', quoting=3)      #IMPORTING STATEMENT

new_dict = {"n't": "not", 'sooooo': 'so', 'soooooo': 'so', 'cant': 'cannot', 'andddd': 'and', 'honeslty': 'honestly',
            'ohhh': 'oh', 'serivce': 'service', 'flavourful': 'flavorful', 'Veggitarian': 'vegetarian',
            'delicioso': 'delicious', 'outta': 'out of', 'transcendant': 'transcendent', 'connisseur': 'connoisseur',
            'absolutley': 'absolutely', 'gooodd': 'good', 'Im': 'I am', 'cavier': 'caviar', 'dont': 'do not',
            'vinegrette': 'vinaigrette ', 'perpared': 'prepared', 'fo': 'of', 'accomodate': 'accommodate',
            'lil': 'little', 'thats': 'that is', 'definately': 'definitely', 'restaraunt': 'restaurant',
            'satifying': 'satisfying', 'pissd': 'pissed', 'devine': 'divine', 'temp': 'temperature',
            'definately': 'definitely', "ya'all": "you all", 'disapppointment': 'disappointment'}
corpus = []

for i in range(0, 1000):
    review = df['Review'][i]
    review = re.sub('[^a-zA-Z!]', ' ', review)
    review = review.lower()
    words = review.split()

    new_word = []
    for word in words:

        if word in new_dict:
            word = new_dict[word]
        new_word.append(word)
    new_text = " ".join(new_word)
    tokens = word_tokenize(new_text)
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    all_stopwords.remove('no')
    all_stopwords.remove('but')
    all_stopwords.remove('won')

    all_stopwords.append('really')
    all_stopwords.append('come')
    all_stopwords.append('get')

    filtered_words = [lem.lemmatize(w, "v") for w in tokens if w not in all_stopwords]
    review = ' '.join(filtered_words)
    corpus.append(review)

# Creating a bag of words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values

# #Splitting the data into testing and training data set
from sklearn.model_selection import train_test_split

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.1, random_state=60, shuffle=True)

# Creating SVM model
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', probability=True, random_state=0)
classifier.fit(x_tr, y_tr)
y_pr = classifier.predict(x_te)

# Finding Accuracy
from sklearn.metrics import plot_confusion_matrix, accuracy_score, balanced_accuracy_score

plot_confusion_matrix(estimator=classifier, X=x_te, y_true=y_te)
acc = accuracy_score(y_te, y_pr)
plt.title('Plotting Confusion Matrix')
plt.show()
print(acc)
