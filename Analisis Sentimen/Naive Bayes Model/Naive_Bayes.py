#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#open dataframe
data = pd.read_excel("Dataset1.xlsx")

#hapus yang null
data["Ulasan_clean"] = data["Ulasan_clean"].fillna("tidak ada komentar")


#tampilkan plot untuk label
sentimen_data=pd.value_counts(data["label"], sort= True)
sentimen_data.plot(kind= 'bar', color= ["green", "red"])
plt.title('Bar chart')
plt.show()


#NLP
#plot ulasan Negatif
train_s0 = data[data["label"] == 0]
print(train_s0)
all_text_s0 = ' '.join(word for word in train_s0["Ulasan_clean"])
wordcloud = WordCloud(colormap='Reds', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s0)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


#plot ulasan positif
train_s1 = data[data["label"] == 1]
all_text_s1 = ' '.join(word for word in train_s1["Ulasan_clean"])
wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s1)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Ulasan Positif")
plt.margins(x=0, y=0)
plt.show()


#menyiapkan train dan test set
X_train, X_test, y_train, y_test = train_test_split(data['Ulasan_clean'], data['label'],
                                                    test_size=0.1, stratify=data['label'], random_state=30)


#TF-IDF
vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train.shape)
print(X_test.shape)
X_train = X_train.toarray()
X_test = X_test.toarray()



cv_method = RepeatedStratifiedKFold(n_splits=5,  n_repeats=3, random_state=999)

nb = GaussianNB()

params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

gscv_nb = GridSearchCV(estimator=nb,
                 param_grid=params_NB,
                 cv=cv_method,   # use any cross validation technique
                 verbose=1,
                 scoring='accuracy')

gscv_nb.fit(X_train,y_train)


#predict model
y_pred_nb = gscv_nb.predict(X_test)
print('confusion matrix')
print(confusion_matrix(y_test, y_pred_nb))
print('classification report')
print(classification_report(y_test, y_pred_nb))