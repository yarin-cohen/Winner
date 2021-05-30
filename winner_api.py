from datetime import  datetime, timedelta
import requests, scipy
import time, re
from bs4 import BeautifulSoup
from selenium import webdriver
import pickle
driver = webdriver.Firefox()


base_url = 'https://www.winner.co.il/results/mishtane?from={}&to={}&category=2&sport=227&league=22989&commit=%D7%94%D7%A6%D7%92'


date_time_format = "%d-%m-%Y"
start_date = datetime.strptime('16-12-2018', date_time_format)
pages = []
while start_date < datetime.now():
    print(start_date)
    url = base_url.format(start_date, start_date + timedelta(days=100))
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    pages.append(soup.find_all("li", {"class": "events result"}))
    start_date+=timedelta(days=7)

date_time_format = "%d.%m.%y"

all_winner_games = []


def proccess_period(game,period):

    def clean_eval(txt):
        for k in range(len(txt)):
            if txt[k] != '.':
                break
        txt = txt[k::]
        return eval(txt)

    if 'points' not in game:
        game["points"]     = []
    if 'advantage' not in game:
        game["advantage"] = []

    if period.find("td", {"class": 'score'}).text == '':
        game['score'] = None
        return

    game['score'] = [int(k) for k in period.find("td", {"class": 'score'}).text.split('-')]
    advantage = [mark for mark in period.find_all("tr", {"class": 'market'}) if mark.find('td',{"class": 'description'}).text == "הימור יתרון"]
    points =  [mark for mark in period.find_all("tr", {"class": 'market'}) if mark.find('td', {"class": 'description'}).text == "מעל/מתחת נקודות"]
    if points != []:
        game["points"] = []
        for d in points[0].find_all('span'):
            txt = re.sub('[^0-9.]', '',d.text.strip())
            game["points"].append(clean_eval(txt))

    if advantage != []:

        for d in advantage[0].find_all('span'):
            game["advantage"].append(clean_eval(re.sub('[^0-9.-]', '', d.text)))



            if game["advantage"][-1] == 0:
                continue
            if game['teams'][1] in d.text:
                game["advantage"][-1]*=-1
            else:
                try:
                    assert (game['teams'][0] in d.text)
                except:
                    game["advantage"][-1] = None


for page in pages:
    for pp in page:
        for header,periods in zip(pp.find_all('h3'),pp.find_all('ul',{"class":"periods"})):
            game = {}
            game['teams'] = [t.strip() for t in header.find("span",{"class","event-description"}).text.split('-') if len(t.strip().split(' '))>1]
            game['date'] = datetime.strptime(header.find('span',{'class':'date'}).text, date_time_format)
            per_ext = [per for per in periods.find_all('li', {'class': 'period'}) if per.find('td',"period-description").text=='כולל הארכות אם יהיו']
            per_no_ext = [per for per in periods.find_all('li', {'class': 'period'}) if
                       per.find('td', "period-description").text == 'ללא הארכות']
            if per_no_ext!=[]:
                proccess_period(game,per_no_ext[0])
            if per_ext != []:
                proccess_period(game,per_ext[0])

            all_winner_games.append(game)


with open('winner_games.pkl','wb+') as file:
    pickle.dump(all_winner_games, file)