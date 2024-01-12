import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#--------------------Model----------------#

data = pd.read_excel("Dataset1.xlsx")


data["Ulasan_clean"] = data["Ulasan_clean"].fillna("tidak ada komentar")


X_train, X_test, y_train, y_test = train_test_split(data['Ulasan_clean'], data['label'],
                                                    test_size=0.1, stratify=data['label'], random_state=30)



vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
X_train = X_train.toarray()
X_test = X_test.toarray()


cv_method = RepeatedStratifiedKFold(n_splits=5,  n_repeats=3, random_state=999)

nb = GaussianNB()

params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

gscv_nb = GridSearchCV(estimator=nb,
                 param_grid=params_NB,
                 cv=cv_method,
                 verbose=1,
                 scoring='accuracy')

gscv_nb.fit(X_train,y_train)


y_pred_nb = gscv_nb.predict(X_test)
print('confusion matrix')
print(confusion_matrix(y_test, y_pred_nb))
print('classification report')
print(classification_report(y_test, y_pred_nb))


#--------------------Plot----------------#
#tampilkan plot untuk label
sentimen_data=pd.value_counts(data["label"], sort= True)
sentimen_data.plot(kind= 'bar', color= ["green", "red"])
plt.title('Sentimen Pengguna')
plt.show()
