import pandas as pd

data = pd.read_excel("Dataset1.xlsx")
text = data['Ulasan_clean']
rating = data['rate']
newdataset = []
#labeling
label = []
for index, row in data.iterrows():
    if row["rate"] == 5 or row["rate"] == 4:
        label.append(1)
    else:
        label.append(0)
data["label"] = label

dataset = {
    "Ulasan_clean" : text,
    "rating" : rating,
    "label" : label
}
df = pd.DataFrame(dataset)

# Menampilkan DataFrame baru
print(df)

df.to_excel("Dataset1.xlsx",index=False)