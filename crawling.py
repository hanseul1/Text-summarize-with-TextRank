from bs4 import BeautifulSoup
from urllib.request import urlopen
import pymongo
from pymongo import MongoClient

# crawling data
url = 'https://search.shopping.naver.com/detail/detail.nhn?nvMid=21462435785&adId=nad-a001-02-000000082496455&channel=nshop.npla&query=%EB%85%B8%ED%8A%B8%EB%B6%81&NaPm=ct%3Dk5j2q6rk%7Cci%3D0zC0000Svg9sOWR2bfnA%7Ctr%3Dpla%7Chk%3D9a9b92e5de6abcf3a47acf5ba973b17346f95380'

html = urlopen(url).read()

soup = BeautifulSoup(html, 'html.parser')
list = soup.find('ul', class_='lst_review')
model = soup.find('div', class_='h_area').find('h2')
reviews = list.find_all('div', class_='atc_area')

data = []
for review in reviews:
    info = review.find_all('span', class_='info_cell')
    reviewData = {
        'title': review.find('p', class_='subjcet').text,
        'writer': info[1].text,
        'model': model.text.strip(),
        'category': 'beauty',
        'rating': review.find('span', class_='curr_avg').text,
        'context': review.find('div', class_='atc').text
    }
    data.append(reviewData)

# MongoDB connection & save data
username = 'hs'
password = '12345'
client = MongoClient('mongodb://%s:%s@localhost:27017/allreview'%(username, password))
db = client['allreview']

result = db.review.insert_many(data)
