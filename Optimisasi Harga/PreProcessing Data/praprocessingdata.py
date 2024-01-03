import pandas as pd
import re

dataframe = pd.read_csv('Tokped_laptop.csv')
Sell = dataframe['sell']
Price = dataframe['price']
Name = dataframe['product_name'].str.lower()
Rating = dataframe['rating']
Location = dataframe['location'].str.lower()
ShopName = dataframe['shop_name'].str.lower()
Category = dataframe['kategori'].str.lower()
dataPrice = []
dataSold = []
dataName = []
dataRating = []
dataLocation = []
dataShopName = []
dataCategory = []
for rows in Price:
    row_num = rows[2:]
    price = re.sub(r'\.', '', row_num)
    dataPrice.append(price)

for rows2 in Sell:
    row_num2 = rows2[:3]
    rb = re.sub(r'rb', '000', row_num2)
    rb1 = re.sub(r'\+', '', rb)
    sell = re.sub(r'r', '000', rb1)
    sell1 = re.sub(r't', '', sell)
    dataSold.append(sell1)

for rows3 in Name:
    name_product = re.findall('asus|advan|acer|hp|lenovo|macbook', rows3)
    dataName.append(name_product)

for rows4 in Rating:
    dataRating.append(rows4)

for rows5 in Location:
    dataLocation.append(rows5)

for rows6 in ShopName:
    dataShopName.append(rows6)

for rows7 in Category:
    dataCategory.append(rows7)

new_dataframe = pd.DataFrame(dataName, columns=['Name'])
new_dataframe['Price'] = dataPrice
new_dataframe['Sold'] = dataSold
new_dataframe['Rating'] = dataRating
new_dataframe['Category'] = dataCategory
new_dataframe['ShopName'] = dataShopName
new_dataframe['Location'] = dataLocation
new_name = new_dataframe['Name']
new_price = new_dataframe['Price']
new_sold = new_dataframe['Sold']
new_rating = new_dataframe['Rating']
new_category = new_dataframe['Category']
new_shopname = new_dataframe['ShopName']
new_location = new_dataframe['Location']
newdata = []
for name, price, sold, rating, category, shopname, location in zip(new_name, new_price, new_sold, new_rating,
                                                                   new_category, new_shopname, new_location):
    if name != None and rating != None and shopname != None and location != None:
        newdata.append((name, price, sold, rating, category, shopname, location))

new_dataframe2 = pd.DataFrame(newdata, columns=['Name', 'Sell', 'Sold', 'Rating', 'Category', 'ShopName', 'Location'])
new_dataframe2.to_csv('Train_Laptop.csv')
print('Done!')

































