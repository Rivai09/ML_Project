import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

#--------------Model------------#
# baca dataset
data = pd.read_excel("Dataset2.xlsx")

# pengisian null
data["Ulasan_clean"] = data["Ulasan_clean"].fillna("tidak ada komentar")

# pisah data
X = data['Ulasan_clean']
y = data['label']

# bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=30)


# TF-IDF
vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Mengonversi ke array numpy
X_train = X_train.toarray()
X_test = X_test.toarray()


# Membuat model menggunakan neural network
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Kompilasi model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# latih model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

#----------------------Visualisasi Model Dan Evaluasi Model--------------------#
# Visualisasi loss dan akurasi
plt.figure(figsize=(8, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot akurasi
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

#plot gant chart label
label_counts = data['label'].value_counts()
plt.figure(figsize=(6, 4))
label_counts.plot(kind='bar', color=['blue', 'red'])
plt.title('Jumlah Data untuk Setiap Label')
plt.xlabel('Label')
plt.ylabel('Jumlah')
plt.xticks(rotation=0)  # Untuk mengatur rotasi label sumbu x
plt.show()

#NLP
#plot ulasan Negatif
train_s0 = data[data["label"] == 0]
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

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy }")

