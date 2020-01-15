from bs4 import BeautifulSoup
from urllib.request import urlopen
import pymongo
from pymongo import MongoClient

# crawling data
url = "https://search.shopping.naver.com/detail/detail.nhn?nv_mid=16658567154&cat_id=50000151&frm=NVSCPRO&query=%EB%85%B8%ED%8A%B8%EB%B6%81&NaPm=ct%3Dk5fcuf6w%7Cci%3De4bc6249377fde0c8384f12441c53b0bb3f36855%7Ctr%3Dslsl%7Csn%3D95694%7Chk%3De4768bf90ea6c0e5d2fc2c763f912d92d4d31ed5"

html = urlopen(url).read()

soup = BeautifulSoup(html, 'html.parser')
list = soup.find('ul', class_='lst_review')

reviews = list.find_all('div', class_='atc')

data = []
for review in reviews:
    r = {
        'context': review.text
    }
    data.append(r)

# MongoDB connection & save data
client = MongoClient()
db = client.allreview
collection = db.review

posts = db.posts
result = posts.insert_many(data)

