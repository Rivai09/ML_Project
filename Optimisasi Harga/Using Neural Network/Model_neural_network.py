#import library
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Train Data to make a prediction about total price based on sold and price

# Preprocessing Data
filename = 'Dataset.csv'
reader = csv.reader(open(filename, "rt"), delimiter=",")
header = next(reader)
sold = []
price = []
rating = []
for row in reader:
    price.append(row[1])
    sold.append(row[2])
    rating.append(row[3])

price = np.array(price, dtype=float)
sold = np.array(sold, dtype=float)
rating = np.array(rating,dtype=float)
PRICE = price / 1000000
SOLD = sold / 100


# Split Data
# Pisahkan fitur dan target
X = PRICE # Fitur: harga dan rating
y = SOLD  # Target: jumlah terjual

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# build model menggunakan neural network
model = tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=[1]),
                    tf.keras.layers.Dense(10),
                    tf.keras.layers.Dense(1)])

# compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3), loss='mean_squared_error',metrics=['mae'])


# train the model
model.fit(X_train, y_train, epochs=100,validation_data=(X_test,y_test))

#make a simple case
inpt = 15
new_x = inpt
prediction = model.predict([new_x])[0][0]
print('Prediksi Jumlah optimal barang yang akan Terjual :',int(prediction * 100) ,'unit' )
# Visualisasi hasil prediksi
plt.scatter(X_test, y_test)
plt.xlabel('harga')
plt.ylabel('jumlah yang terjual')
plt.title('Hubungan Harga Produk dan Jumlah Penjualan')
plt.show()
