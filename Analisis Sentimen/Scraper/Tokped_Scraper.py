import time
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
inputan = input("silahkan masukkan data yang ingin dicari di tokopedia : ")
url = "https://www.tokopedia.com/search?st=&q="+inputan+"&srp_component_id=02.01.00.00&srp_page_id=&srp_page_title=&navsource="
driver = webdriver.Chrome()
driver.get(url)

data=[]

for i in range(1):
 WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#zeus-root')))
 time.sleep(5)
 for j in range(35):
  driver.execute_script("window.scrollBy(0,250)")
  time.sleep(5)
 soup = BeautifulSoup(driver.page_source, 'html.parser')
 driver.execute_script("window.scrollBy(50,0)")
 time.sleep(5)

 #ambil nama produk dan harga
 for item in soup.findAll('div',class_="css-1asz3by"):
  product_name = item.find('div',class_="prd_link-product-name css-3um8ox").text
  price        = item.find('div',class_="prd_link-product-price css-h66vau").text


  #ambil rating dan terjual
  for item2 in item.findAll('div',class_="prd_shop-rating-average-and-label css-26zmlj"):
   try:
    rating = item.find('span', class_="prd_rating-average-text css-t70v7i").text
   except:
    rating = None
   try:
    sell = item.find('span', class_="prd_label-integrity css-1sgek4h").text
   except:
    sell = None
  for item3 in item.findAll('div',class_='css-1rn0irl'):
   try:
    location = item.find('span',class_='prd_link-shop-loc css-1kdc32b flip').text
   except:
    location = None
   try:
    shop_name = item.find('span', class_='prd_link-shop-name css-1kdc32b flip').text
   except:
    shop_name = None

   data.append((product_name, price, rating, sell, location, shop_name))



df = pd.DataFrame(data, columns=['product_name','price','rating','sell','location','shop_name'])
df['kategori'] = inputan
df.to_csv('Tokped_'+inputan+'.csv',index=False)
df = pd.read_csv('Tokped_'+inputan+'.csv')
df = df.dropna()
df.to_csv('Tokped_'+inputan+'.csv',index=False)
print("data "+inputan+" berhasil disimpan")

driver.close()
















'''
data=[]

for item in soup.findAll('div',class_="css-1asz3by"):
 product_name = item.find('div',class_="prd_link-product-name css-3um8ox").text
 price        = item.find('div',class_="prd_link-product-price css-h66vau").text

 rtg = item.find('span', class_="prd_rating-average-text css-t70v7i")
 if len(rtg) > 0:
  rating = item.find('span', class_="prd_rating-average-text css-t70v7i").text
 else:
  rating = ''

 sel= item.find('span',class_="prd_label-integrity css-1sgek4h")
 if len(sel) > 0:
  sell = item.find('span',class_="prd_label-integrity css-1sgek4h").text
 else:
  sel = ''

data.append((product_name,price,rating,sell))

df = pd.DataFrame(data, columns=['product_name','price','rating','sell'])
print(df)
df.to_excel('Tokped_.xlsx',index=False)
print("data berhasil disimpan")'''


