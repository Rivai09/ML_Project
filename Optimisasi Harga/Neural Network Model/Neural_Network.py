#import library
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

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
tempprice = []
tempsold = []
temprating= []

for i,j,k in zip(price,sold,rating):
    integerprice  = int(i)
    integersold   = int(j)
    floatrating = float(k)
    tempprice.append(integerprice)
    tempsold.append(integersold)
    temprating.append(floatrating)


PRICE = np.array(tempprice).astype(float)
SOLD = np.array(tempsold).astype(float)
RATING = np.array(temprating).astype(float)


# Split Data
# Pisahkan fitur dan target
X = PRICE / 1000000
y = SOLD / 100



# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# build model menggunakan neural network
model = tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=[1]),
                    tf.keras.layers.Dense(10),
                    tf.keras.layers.Dense(1)])

# compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3), loss='mean_squared_error',metrics=['mae'])



# train the model
history = model.fit(X_train, y_train, epochs=100,batch_size=32,validation_data=(X_test,y_test))

#make a simple case
inpt = 15
new_x = inpt
prediction = model.predict([new_x])[0][0]
print('Prediksi Jumlah rata-rata barang yang akan Terjual :',int(prediction * 100) ,'unit' )

#evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy }")
