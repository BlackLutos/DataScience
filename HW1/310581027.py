import argparse
import requests
import json
from bs4 import BeautifulSoup
import re
import time

def crawl():
    all_articles = []
    page_num = 3647 # 2022 Year from 3647 to 3955
    start_time = time.time()
    while page_num <= 3955:
        print("Page. " + str(page_num) + "\n")
        query_url = "https://www.ptt.cc/bbs/Beauty/index" + str(page_num) + ".html"
        response = requests.get(query_url, cookies={"over18": "1"})
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("div", {"class": "r-ent"})
        for article in articles:
            title = article.find("div", {"class": "title"}).text.strip()
            meta = article.find("div", {"class": "meta"})
            date = meta.find("div", {"class": "date"}).text.strip()
            if not re.match("(Fw: )?\[公告\]", title):
                href = article.find("a")
                if href:
                    url = "https://www.ptt.cc" + href.get("href")
                    all_articles.append({"date": date, "title": title, "url": url})
                    print(date + ":" + title + " " + url)
        time.sleep(1)
        print("\n")
        page_num += 1

    del all_articles[0:2]
    del all_articles[-4:]
    with open("all_articles.jsonl", "w", encoding="utf8") as f:
        for article in all_articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
    
    print("Total time: " + str(time.time() - start_time) + " seconds")

def push():
    print("Pushing...")

def popular():
    print("Popular...")

def keyword():
    print("Keyword...")



parser = argparse.ArgumentParser()
parser.add_argument("arg1")
parser.add_argument("arg2", nargs="?",default="NO_VALUE")
parser.add_argument("arg3", nargs="?",default="NO_VALUE")
parser.add_argument("arg4", nargs="?",default="NO_VALUE")
args = parser.parse_args()

if args.arg1 == "crawl":
    crawl()
elif args.arg1 == "push":
    push()
elif args.arg1 == "popular":
    popular()
elif args.arg1 == "keyword":
    keyword()
else:
    print("Invalid command")


