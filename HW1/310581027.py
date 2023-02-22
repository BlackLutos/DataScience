import argparse
import requests
import json
from bs4 import BeautifulSoup
import re
import time

def crawl():
    all_articles = []
    page_num = 3647 # 2022 Year from 3647 to 3955
    while page_num <= 3955:
        print("Page. " + str(page_num) + "\n")
        query_url = "https://www.ptt.cc/bbs/Beauty/index" + str(page_num) + ".html"
        response = requests.get(query_url, cookies={"over18": "1", "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"})
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
    

def push(start_date, end_date):
    push_like = []
    push_boo = []
    all_articles = []
    old_start_date = start_date
    old_end_date = end_date
    start_date = start_date[0:2] + '/' + start_date[2:4]
    end_date = end_date[0:2] + '/' + end_date[2:4]
    time_limit = 0
    with open("all_articles.jsonl", "r", encoding="utf8") as f:
        all_articles = [json.loads(line) for line in f]
    for article in all_articles:
        if article['date'] >= start_date and article['date'] <= end_date:
            url = article["url"]
            response = requests.get(url, cookies={"over18": "1", "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"})
            soup = BeautifulSoup(response.text, "html.parser")
            pushs = soup.find_all("div", {"class": "push"})
            push_count = 0
            boo_count = 0
            for push in pushs:
                push_tag = push.find("span", {"class": "push-tag"}).text.strip()
                user_id = push.find("span", {"class": "f3 hl push-userid"}).text.strip()
                if "推" in push_tag:
                    push_count += 1
                    push_like.append({"user_id": user_id, "like": 1})   
                elif "噓" in push_tag:
                    boo_count += 1
                    push_boo.append({"user_id": user_id, "boo": 1})
            article["push_count"] = push_count
            article["boo_count"] = boo_count
            # print(article["date"] + ":" + article["title"] + " 推:" + str(push_count) + " 噓:" + str(boo_count))
            if time_limit % 100 == 0:
                time.sleep(1)
            time_limit += 1

    like_count = {}
    for user in push_like:
        if user["user_id"] in like_count:
            like_count[user["user_id"]] += 1
        else:
            like_count[user["user_id"]] = 1
    like_counter = sum(like_count.values())
    like_count = dict(sorted(like_count.items(), key=lambda x: (x[1],x[0]), reverse=True))
    boo_count = {}
    for user in push_boo:
        if user["user_id"] in boo_count:
            boo_count[user["user_id"]] += 1
        else:
            boo_count[user["user_id"]] = 1
    boo_counter = sum(boo_count.values())
    boo_count = dict(sorted(boo_count.items(), key=lambda x: (x[1],x[0]), reverse=True))
    push = {"all-like": like_counter, "all-boo": boo_counter}
    like_num = 1
    boo_num = 1
    for key in list(like_count.keys())[:10]:
        push.update({"like " + str(like_num): {"user_id":key, "count": like_count[key]}})
        like_num += 1
    for key in list(boo_count.keys())[:10]:
        push.update({"boo " + str(boo_num): {"user_id":key, "count": boo_count[key]}})
        boo_num += 1
    print(push)
    filename = "push_" + old_start_date + "_" + old_end_date + ".json"
    with open(filename, "w", encoding="utf8") as f:
        json.dump(push, f, ensure_ascii=False)

def popular():
    print("Popular...")

def keyword():
    print("Keyword...")

if __name__ == "__main__":

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("arg1")
    parser.add_argument("arg2", nargs="?",default="NO_VALUE")
    parser.add_argument("arg3", nargs="?",default="NO_VALUE")
    parser.add_argument("arg4", nargs="?",default="NO_VALUE")
    args = parser.parse_args()

    if args.arg1 == "crawl":
        crawl()
    elif args.arg1 == "push":
        push(str(args.arg2), str(args.arg3))
    elif args.arg1 == "popular":
        popular()
    elif args.arg1 == "keyword":
        keyword()
    else:
        print("Invalid command")

    print("Total time: " + str(time.time() - start_time) + " seconds")


