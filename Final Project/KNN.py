import numpy as np
import pandas as pd

# Import dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')

import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review
              if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)

X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Fitting Random Forest Classification  to the Training set
from sklearn.ensemble import RandomForestClassifier

# n_estimators can be said as number of

model = RandomForestClassifier(n_estimators=501,
                               criterion='entropy')

model.fit(X_train, y_train)

from sklearn.metrics import plot_confusion_matrix, confusion_matrix, accuracy_score

Confusion_Matrix = confusion_matrix(y_test, y_pred)
Accuracy_Score = accuracy_score(y_test, y_pred)
# plotting and printing accuracy
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
df_cm = pd.DataFrame(Confusion_Matrix, range(2),
                     range(2))

sns.set(font_scale=1)  # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 17})  # font size
print("Accuracy Score is :", Accuracy_Score * 100)
plt.title('Confussion Matrix')
plt.show()
