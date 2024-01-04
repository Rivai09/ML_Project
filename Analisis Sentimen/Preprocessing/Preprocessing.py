import pandas as pd
import re
#buka dataset
data = pd.read_csv("Dataset.csv")
text = data['text']
rating = data['rating']
name = data['product_name']
label = []
print(name)

#labeling 1 untuk rating 5/4(positif), 0 untuk 3 kebawah(negatif)
for index, row in data.iterrows():
    if row["rating"] == 5 or row["rating"] == 4:
        label.append(1)
    else:
        label.append(0)
data["label"] = label

# Menampilkan DataFrame baru
dataset = {
    "review": text,
    "rating": rating,
    "label": label
}

df = pd.DataFrame(dataset)
#convert dataframe
df.to_excel("Dataset1.xlsx",index=False)
