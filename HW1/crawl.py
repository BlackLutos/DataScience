import requests
import json
from bs4 import BeautifulSoup
import re
import time

def test():
    a = [1,2,3,4,5,6,7,8,9,10]
    del a[0:2]
    del a[-4:]
    print(a)


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

if __name__ == "__main__":
    crawl()
