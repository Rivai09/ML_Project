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
data = pd.read_excel("tokped_text.xlsx", index_col=0)

#labeling
label = []
for index, row in data.iterrows():
    if row["rate"] == 5 or row["rate"] == 4:
        label.append(1)
    else:
        label.append(0)
data["label"] = label
data_label = data[["Nama_Produk", "Akun", "Ulasan_clean", "label"]]
data_label["Ulasan_clean"] = data_label["Ulasan_clean"].fillna("tidak ada komentar")
#data_label.to_excel("data_label.xlsx")


#tampilkan plot untuk label
sentimen_data=pd.value_counts(data_label["label"], sort= True)
sentimen_data.plot(kind= 'bar', color= ["green", "red"])
plt.title('Bar chart')
plt.show()


#plot ulasan Negatif
train_s0 = data_label[data_label["label"] == 0]
train_s0["Ulasan_clean"] = train_s0["Ulasan_clean"].fillna("tidak ada komentar")
print(train_s0)
all_text_s0 = ' '.join(word for word in train_s0["Ulasan_clean"])
wordcloud = WordCloud(colormap='Reds', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s0)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


#plot ulasan positif
train_s1 = data_label[data_label["label"] == 1]
train_s1["Ulasan_clean"] = train_s1["Ulasan_clean"].fillna("tidak ada komentar")
all_text_s1 = ' '.join(word for word in train_s1["Ulasan_clean"])
wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s1)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Ulasan Positif")
plt.margins(x=0, y=0)
plt.show()


#menyiapkan train dan test set
data_label['Ulasan_clean'] = data_label['Ulasan_clean'].fillna("tidak ada komentar")
X_train, X_test, y_train, y_test = train_test_split(data_label['Ulasan_clean'], data_label['label'],
                                                    test_size=0.1, stratify=data_label['label'], random_state=30)


#TF-IDF
vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train.shape)
print(X_test.shape)
X_train = X_train.toarray()
X_test = X_test.toarray()


#machine learning(gaussian)

class GaussianNaiveBayes:
    def __init__(self,smoothing=1e-9):
        self.class_prob = {}  # Probabilitas prior untuk setiap kelas
        self.mean = {}  # Rata-rata fitur untuk setiap kelas
        self.var = {}  # Varians fitur untuk setiap kelas

    def fit(self, X, y):
        # Hitung probabilitas prior untuk setiap kelas
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        for cls, count in zip(classes, counts):
            self.class_prob[cls] = count / total_samples

        # Hitung rata-rata dan varians fitur untuk setiap kelas
        for cls in classes:
            X_cls = X[y == cls]
            self.mean[cls] = np.mean(X_cls, axis=0)
            self.var[cls] = np.var(X_cls, axis=0)

    def predict(self, X):
        preds = []
        for sample in X:
            # Hitung probabilitas untuk setiap kelas
            probs = {}
            for cls in self.class_prob:
                prior = np.log(self.class_prob[cls])
                probs[cls] = prior + np.sum(-0.5 * np.log(2 * np.pi * self.var[cls]) -
                                            (sample - self.mean[cls]) ** 2 / (2 * self.var[cls]))
            # Pilih kelas dengan probabilitas tertinggi sebagai prediksi
            pred_class = max(probs, key=probs.get)
            preds.append(pred_class)
        return preds


#deklarasi metode cross validation
cv_method = RepeatedStratifiedKFold(n_splits=5,  n_repeats=3, random_state=999)

#panggil class naive bayes
nb2 = GaussianNaiveBayes(smoothing=1e-9)

#panggil naive bayes menggunakan skcitlearn
nb = GaussianNB()

#tuning hyperparameter menggunakan gridsearch
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}

gscv_nb = GridSearchCV(estimator=nb,
                 param_grid=params_NB,
                 cv=cv_method,   # use any cross validation technique
                 verbose=1,
                 scoring='accuracy')

#Fitting ke Model
gscv_nb.fit(X_train,y_train)

#mendapatkan hyperparameters terbaik
var = gscv_nb.best_params_

#Fitting ke model dengan parameter smoothing
nb = GaussianNB(var_smoothing=1.0)
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
print('--------------------- confusion matrix  ----------------------------')
print(confusion_matrix(y_test, y_pred_nb))
print('--------------------- classification report  ----------------------------')
print(classification_report(y_test, y_pred_nb))