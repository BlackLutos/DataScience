import argparse
import requests
import json
from bs4 import BeautifulSoup
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from threading import Thread
import os

def crawl():
    all_articles = []
    all_popular = []
    # article_num = 0
    # first_article = 0
    # last_article = 0
    # popular_num = 0
    # popular_first = 0
    # popular_last = 0
    article_flag = 0
    page_num = 3600 # 3642
    end_page = 3951
    while page_num <= end_page:
        print("Scan Page. " + str(page_num) + "\n")
        query_url = "https://www.ptt.cc/bbs/Beauty/index" + str(page_num) + ".html"
        response = requests.get(query_url, cookies={"over18": "1", "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"})
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("div", {"class": "r-ent"})
        for article in articles:
            title = article.find("div", {"class": "title"}).text.strip()
            meta = article.find("div", {"class": "meta"})
            date = meta.find("div", {"class": "date"}).text.strip()
            popular = article.find("div", {"class": "nrec"}).text.strip()
            if len(date) == 4:
                date = str(0) + date
            if not re.match("(Fw: )?\[公告\]", title):
                # article_num += 1
                # if "[帥哥] 羅志祥強勢回歸" in title:
                #     article_flag = 1
                href = article.find("a")
                if href:
                    url = "https://www.ptt.cc" + href.get("href")
                    all_articles.append({"date": date, "title": title, "url": url})
                    # print(date + ":" + title + " " + url)
                    if popular == "爆":
                        # print(date + ":" + title + " " + url)
                        all_popular.append({"date": date, "title": title, "url": url})
                # if "[正妹] 孟潔MJ" in title:
                #     article_flag = 0
            
                
        time.sleep(1)
        page_num += 1
    
    all_articles_new = []
    First_half_articles = all_articles[:len(all_articles)//2]
    First_half_date = First_half_articles[-1]["date"]
    Second_half_articles = all_articles[len(all_articles)//2:]
    for article in First_half_articles:
        if article['date'] >= "01/01" and article['date'] <= First_half_date:
                all_articles_new.append(article)
    for article in Second_half_articles:
        if article['date'] >= First_half_date and article['date'] <= "12/31":
                all_articles_new.append(article)

    all_popular_new = []
    First_half_popular = all_popular[:len(all_popular)//2]
    First_half_date = First_half_popular[-1]["date"]
    Second_half_popular = all_popular[len(all_popular)//2:]
    for article in First_half_popular:
        if article['date'] >= "01/01" and article['date'] <= First_half_date:
                all_popular_new.append(article)
    for article in Second_half_popular:
        if article['date'] >= First_half_date and article['date'] <= "12/31":
                all_popular_new.append(article)
    


    with open("all_articles.jsonl", "w", encoding="utf8") as f:
        for article in all_articles_new:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
    with open("all_popular.jsonl", "w", encoding="utf8") as f:
        for popular in all_popular_new:
            f.write(json.dumps(popular, ensure_ascii=False) + "\n")
    
def merge_dict(push_like, push_boo, start_date, end_date):
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
    filename = "push_" + start_date + "_" + end_date + ".json"
    with open(filename, "w", encoding="utf8") as f:
        json.dump(push, f, ensure_ascii=False)

def thread_push(start_date, end_date):
    start_time = time.time()

    date1_obj = datetime.strptime(start_date, '%m%d').replace(year=datetime.now().year)
    date2_obj = datetime.strptime(end_date, '%m%d').replace(year=datetime.now().year)

    
    delta = (date2_obj - date1_obj) / 2
    middle_date = date1_obj + delta
    next_date = middle_date + timedelta(days=1)
    
    middle_date_str = middle_date.strftime('%m%d')
    next_date_str = next_date.strftime('%m%d')

   
    t1 = Thread (target = push, args =(start_date,middle_date_str,))
    t1.start()
    t2 = Thread (target = push, args =(next_date_str,end_date,))
    t2.start()
    t1.join()
    t2.join()

    push_like_prev_filename = "push_like_" + start_date + "_" + middle_date_str + ".json"
    push_boo_prev_filename = "push_boo_" + start_date + "_" + middle_date_str + ".json"
    push_like_next_filename = "push_like_" + next_date_str + "_" + end_date + ".json"
    push_boo_next_filename = "push_boo_" + next_date_str + "_" + end_date + ".json"

    with open(push_like_prev_filename, "r") as f:
        push_like = json.load(f)
    with open(push_boo_prev_filename, "r") as f:
        push_boo = json.load(f)
    with open(push_like_next_filename, "r") as f:
        push_like_next = json.load(f)
    with open(push_boo_next_filename, "r") as f:
        push_boo_next = json.load(f)

    push_like.extend(push_like_next)
    push_boo.extend(push_boo_next)

    merge_dict(push_like, push_boo, start_date, end_date)

    os.remove(push_like_prev_filename)
    os.remove(push_boo_prev_filename)
    os.remove(push_like_next_filename)
    os.remove(push_boo_next_filename)

def push(start_date, end_date):
    push_like = []
    push_boo = []
    temp_articles = []
    all_articles = []
    old_start_date = start_date
    old_end_date = end_date
    start_date = start_date[0:2] + '/' + start_date[2:4]
    end_date = end_date[0:2] + '/' + end_date[2:4]
    time_limit = 0
    with open("all_articles.jsonl", "r", encoding="utf8") as f:
        temp_articles = [json.loads(line) for line in f]
    for i in temp_articles:
        if i["date"] >= start_date and i["date"] <= end_date:
            all_articles.append(i)
    for article in all_articles:
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
        if time_limit % 100 == 0:
            time.sleep(1)
        time_limit += 1
    push_like_filename = "push_like_" + old_start_date + "_" + old_end_date + ".json"
    push_boo_filename = "push_boo_" + old_start_date + "_" + old_end_date + ".json"
    with open(push_like_filename, 'w', encoding="utf8") as f:
        json.dump(push_like, f)
    with open(push_boo_filename, 'w', encoding="utf8") as f:
        json.dump(push_boo, f)
    print(start_date, end_date, "done")

def popular(start_date, end_date):
    push_like = []
    push_boo = []
    all_articles = []
    old_start_date = start_date
    old_end_date = end_date
    start_date = start_date[0:2] + '/' + start_date[2:4]
    end_date = end_date[0:2] + '/' + end_date[2:4]
    time_limit = 0
    popular_article = {"number_of_popular_articles": 0}
    img_url = []
    with open("all_popular.jsonl", "r", encoding="utf8") as f:
        all_articles = [json.loads(line) for line in f]
    for article in all_articles:
        if article['date'] >= start_date and article['date'] <= end_date:
            url = article["url"]
            response = requests.get(url, cookies={"over18": "1", "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"})
            soup = BeautifulSoup(response.text, "html.parser")
            pushs = soup.find_all("div", {"class": "push"})
            push_count = 0
            # boo_count = 0
            for push in pushs:
                push_tag = push.find("span", {"class": "push-tag"}).text.strip()
                user_id = push.find("span", {"class": "f3 hl push-userid"}).text.strip()
                if "推" in push_tag:
                    push_count += 1
                    push_like.append({"user_id": user_id, "like": 1})
            if push_count >= 100:
                imgs = soup.findAll('a', {'href': re.compile('(https:|http:).*\.(jpg|jpeg|png|gif)$')})
                url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
                imgs = url_pattern.findall(str(imgs))
                step = 1
                for img in imgs:
                    if step % 2 != 0:
                        img_url.append(img)
                    step += 1
                popular_article['number_of_popular_articles'] += 1
            if time_limit % 100 == 0:
                time.sleep(1)
            time_limit += 1
    popular_article.update({"image_urls": img_url})
    # print("Total push article: ", popular_article)
    filename = "popular_" + old_start_date + "_" + old_end_date + ".json"
    with open(filename, "w", encoding="utf8") as f:
        json.dump(popular_article, f, ensure_ascii=False)

def keyword(key_word, start_date, end_date):

    temp_articles = []
    all_articles = []
    keyword_article = {}
    old_start_date = start_date
    old_end_date = end_date
    start_date = start_date[0:2] + '/' + start_date[2:4]
    end_date = end_date[0:2] + '/' + end_date[2:4]
    time_limit = 0
    img_url = []
    with open("all_articles.jsonl", "r", encoding="utf8") as f:
        temp_articles = [json.loads(line) for line in f]
    for i in temp_articles:
        if key_word in i['title'] and i['date'] >= start_date and i['date'] <= end_date:
            all_articles.append(i)
            # print(i)
    for article in all_articles:
        url = article["url"]
        response = requests.get(url, cookies={"over18": "1", "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"})
        soup = BeautifulSoup(response.text, "html.parser")
        imgs = soup.findAll('a', {'href': re.compile('(https:|http:).*\.(jpg|jpeg|png|gif)$')})
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        imgs = url_pattern.findall(str(imgs))
        step = 1
        for img in imgs:
            if step % 2 != 0:
                img_url.append(img)
            step += 1
        if time_limit % 100 == 0:
                time.sleep(1)
        time_limit += 1
        # print(img_url)
    key_word_filename = "keyword_" + key_word + "_" + old_start_date + "_" + old_end_date + ".json"
    with open(key_word_filename, "w", encoding="utf8") as f:
        json.dump(img_url, f, ensure_ascii=False)

    # keyword_article.update({"image_urls": img_url})
    # filename = "keyword_" + key_word + "_" + old_start_date + "_" + old_end_date + ".json"
    # with open(filename, "w", encoding="utf8") as f:
    #     json.dump(keyword_article, f, ensure_ascii=False)

    print(start_date, end_date, "done")

def merge_keyword(img_url_list1,img_url_list2, key_word, start_date, end_date):
    keyword_article = {}
    img_url_list1.extend(img_url_list2)
    keyword_article.update({"image_urls": img_url_list1})
    filename = "keyword_" + key_word + "_" + start_date + "_" + end_date + ".json"
    with open(filename, "w", encoding="utf8") as f:
        json.dump(keyword_article, f, ensure_ascii=False)

def thread_keyword(key_word ,start_date, end_date):
    start_time = time.time()

    date1_obj = datetime.strptime(start_date, '%m%d').replace(year=datetime.now().year)
    date2_obj = datetime.strptime(end_date, '%m%d').replace(year=datetime.now().year)

    
    delta = (date2_obj - date1_obj) / 2
    middle_date = date1_obj + delta
    next_date = middle_date + timedelta(days=1)
    
    middle_date_str = middle_date.strftime('%m%d')
    next_date_str = next_date.strftime('%m%d')

   
    t1 = Thread (target = keyword, args =(key_word,start_date,middle_date_str,))
    t1.start()
    t2 = Thread (target = keyword, args =(key_word, next_date_str,end_date,))
    t2.start()
    t1.join()
    t2.join()

    img_url_list1_filename = "keyword_" + key_word + "_" + start_date + "_" + middle_date_str + ".json"
    img_url_list2_filename = "keyword_" + key_word + "_" + next_date_str + "_" + end_date + ".json"

    with open(img_url_list1_filename, "r", encoding="utf8") as f:
        img_url_list1 = json.load(f)
    with open(img_url_list2_filename, "r", encoding="utf8") as f:
        img_url_list2 = json.load(f)

    merge_keyword(img_url_list1,img_url_list2, key_word, start_date, end_date)

    os.remove(img_url_list1_filename)
    os.remove(img_url_list2_filename)

    
    

if __name__ == "__main__":

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("arg1")
    parser.add_argument("arg2", nargs="?",default="NO_VALUE")
    parser.add_argument("arg3", nargs="?",default="NO_VALUE")
    parser.add_argument("arg4", nargs="?",default="NO_VALUE")
    args = parser.parse_args()

    if args.arg1 == "crawl":
        print("Start crawling...")
        crawl()
    elif args.arg1 == "push":
        # push(str(args.arg2), str(args.arg3))
        print("Start pushing...")
        thread_push(str(args.arg2), str(args.arg3))
    elif args.arg1 == "popular":
        print("Start popular...")
        popular(str(args.arg2), str(args.arg3))
    elif args.arg1 == "keyword":
        print("Strat keyword...")
        thread_keyword(str(args.arg2), str(args.arg3), str(args.arg4))
    else:
        print("Invalid command")

    print("Total time: " + str(time.time() - start_time) + " seconds")


