import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()


df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

new_dict = {"n't":"not", 'sooooo': 'so', 'soooooo': 'so', 'cant': 'cannot','andddd':'and', 'honeslty' : 'honestly', 'ohhh': 'oh','serivce': 'service','flavourful' : 'flavorful', 'Veggitarian' : 'vegetarian', 'delicioso': 'delicious','outta' : 'out of', 'transcendant': 'transcendent', 'connisseur': 'connoisseur' , 'absolutley': 'absolutely','gooodd':'good', 'Im': 'I am', 'cavier' : 'caviar', 'dont':'do not', 'vinegrette': 'vinaigrette ','perpared': 'prepared', 'fo':'of', 'accomodate': 'accommodate', 'lil': 'little', 'thats': 'that is','definately': 'definitely','restaraunt': 'restaurant', 'satifying': 'satisfying', 'pissd': 'pissed', 'devine' :'divine', 'temp': 'temperature','definately':'definitely',"ya'all":"you all", 'disapppointment': 'disappointment' }
corpus = []

for i in range(0,1000):
    review = df['Review'][i]
    review = re.sub('[^a-zA-Z!]', ' ', review)
    review = review.lower()
    words = review.split()
    # print(words)
    new_word= []
    for word in words:
        #print(word)
        if word in new_dict:
            word = new_dict[word]
        new_word.append(word) # a new list of standard words
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

# Creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=None, copy = True, whiten= False)
x = pca.fit_transform(x)
var = pca.explained_variance_ratio_.cumsum()


# #Splitting into test and training set
from sklearn.model_selection import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(x,y,test_size=0.1,random_state=60, shuffle=True) #Changed

# Creating SVM model
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', probability= True ,random_state=0) #Changed
classifier.fit(x_tr, y_tr)
y_pr = classifier.predict(x_te)

# Finding Accuracy
from sklearn.metrics import plot_confusion_matrix, accuracy_score, balanced_accuracy_score

# Plotting Confusion Matrix
plot_confusion_matrix(estimator=classifier, X=x_te, y_true=y_te)
plt.title('Plotting Confusion Matrix')
plt.show()

# Plotting variance(Eigen values) to find the best N_estimator for PCA.

plt.bar(x = range(1,len(var)+1),height =var, width = 0.1)
plt.title('Plotting variance to find the best suitable N_estimator for PCA')
plt.show()

#Printing Accuracy Score

acc = accuracy_score(y_te,y_pr)
print(acc)
