#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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


#menyiapkan train dan test set
data['Ulasan_clean'] = data['Ulasan_clean'].fillna("tidak ada komentar")
X_train, X_test, y_train, y_test = train_test_split(data['Ulasan_clean'], data['label'],
                                                    test_size=0.1, stratify=data['label'], random_state=30)


#TF-IDF
vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
X_train = X_train.toarray()
X_test = X_test.toarray()


#machine learning(class gaussian)
class GaussianNaiveBayes:
    def __init__(self, smoothing=1e-9):
        self.smoothing = smoothing
        self.class_prob = {}
        self.mean = {}
        self.var = {}

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
            self.var[cls] = np.var(X_cls, axis=0) + self.smoothing

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


#panggil class naive bayes
nb = GaussianNaiveBayes()

#Fitting ke model menggunakan class naive bayes
history = nb.fit(X_train, y_train)



y_pred_nb = nb.predict(X_test)
print('--------------------- confusion matrix  ----------------------------')
print(confusion_matrix(y_test, y_pred_nb))
print('--------------------- classification report  ----------------------------')
print(classification_report(y_test, y_pred_nb))