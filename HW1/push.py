import requests
import json
from bs4 import BeautifulSoup
import re
import time
from collections import Counter

def test():
    x = {'a': 1, 'b': 2, 'c': 3}
    a = []
    a.append(x)
    a.append(x)
    print(a)

def push():
    push_like = []
    push_boo = []
    with open("all_articles.jsonl", "r", encoding="utf8") as f:
        all_articles = [json.loads(line) for line in f]
    for article in all_articles[:15]:
        url = article["url"]
        response = requests.get(url, cookies={"over18": "1"})
        soup = BeautifulSoup(response.text, "html.parser")
        pushs = soup.find_all("div", {"class": "push"})
        push_count = 0
        boo_count = 0
        for push in pushs:
            push_tag = push.find("span", {"class": "push-tag"}).text.strip()
            user_id = push.find("span", {"class": "f3 hl push-userid"}).text.strip()
            # print("user_id" + user_id)
            if "推" in push_tag:
                push_count += 1
                push_like.append({"user_id": user_id, "like": 1})   
            elif "噓" in push_tag:
                boo_count += 1
                push_boo.append({"user_id": user_id, "boo": 1})
        article["push_count"] = push_count
        article["boo_count"] = boo_count
        print(article["date"] + ":" + article["title"] + " " + str(push_count) + " " + str(boo_count))
        time.sleep(1)

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
    print("like_count")
    print(like_count)
    print("like_counter", like_counter)
    print("boo_count")
    print("boo_counter", boo_counter)
    print(boo_count)
    print("push")
    push = {"all_like": like_counter, "all_boo": boo_counter}
    print(push)



if __name__ == "__main__":
    push()