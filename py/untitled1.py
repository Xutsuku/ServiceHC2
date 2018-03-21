# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup


def get_cities():
    url = 'http://www.wenku1.com/news/0D3FCAB879060BCB.html'
    html = requests.get(url)
    bsobj = BeautifulSoup(html.text, "lxml")
    txt = bsobj.find("div", {"id": "doctext"}).find_all('p')
    cities = []
    for city in txt:
        city = str(city)
        if "<p>" in city and " " in city:
            city = str(city).strip("<p></p >")
            cities.append(city.split(' '))
    # 简单清洗数据
    cities = [i for i in cities if len(i)==3]
    return cities


def find_one_road(city):
    cities = get_cities()
    res = []
    for i in cities:
        if i[0] == city:
            del cities[cities.index(i)]
            res.append(i)
            break
    def find_next():
        last_char = res[-1][2][-1]
        for i in cities:
            if i[2].startswith(last_char):
                res.append(i)
                del cities[cities.index(i)]
                return True
        return False
    while True:
        matched = find_next()
        if not matched:
            break
    return res
                
if __name__ == "__main__":
    find_one_road(city="济南")