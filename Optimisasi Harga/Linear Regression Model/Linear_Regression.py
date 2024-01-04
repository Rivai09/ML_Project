import numpy as np
import csv
import matplotlib.pyplot as plt

filename = 'Dataset.csv'
reader = csv.reader(open(filename, "rt"), delimiter=",")
header = next(reader)
sold = []
price = []
for row in reader:
    price.append(row[1])
    sold.append(row[2])

price = np.array(price, dtype=float)
sold = np.array(sold, dtype=float)
PRICE = price / 1000000
SOLD = sold / 100
mean_X = np.mean(PRICE)
mean_Y = np.mean(SOLD)

m = len(price)

numer = 0
denom = 0
for i in range(m):
    numer += (price[i] - mean_X) * (sold[i] - mean_Y)
    denom += (price[i] - mean_X)**2

m = numer/denom
c = mean_Y - (m * mean_X)

max_X = np.max(price) + 100
min_X = np.min(sold) - 100

X = np.linspace(min_X, max_X, 500)
Y = c + m * X

plt.plot(X,Y)
plt.scatter(price, sold, color="#00FF00", label="data")
plt.title('Persebaran Data')
plt.xlabel("Harga")
plt.ylabel("Terjual")

plt.legend()
plt.show()