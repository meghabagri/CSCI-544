import sys
import os
from string import digits
import string
import json
import math
import re


def print_model(priors, cond_prob, vocab_size):
    output_file = "nbmodel.txt"

    with open(output_file, "w") as outfile:
        outfile.write(json.dumps(
            {"prior_prob": priors, "cond_prob": cond_prob, "vocab_size": vocab_size}, indent=4))


def cleanup(review):
    stop_words = {'here', 'few', 'last', 'week', 'night', 'nights', 'rooms', 'room', 'chicago', 'arrived', 'staff', 'bed', 'shower', 'lobby', 'kitchen', 'downtown', 'put', 'putting', 'tub', 'ave', 'avenue', 'loaction', 'view', 'hotel', 't','upto', 'but', 'asked', 'out', 'her', 'by', 'theirs', 'been', 'itself', 'if', 'themselves', 'each', 'she', 'very', 'what', 'than', 'then', 'having', 'some', 'do', 'such', 'who', 'nor', 'does', 'from', 'doing', 'just', 'theyll', 'you', 'or', 'both', 'as', 'no', 'hey', 'on', 'below', 'which', 'be', 'hi', 'so', 'this', 'too', 'in', 'before', 'said', 'over', 'after', 'its','between', 'my', 'ill', 'at', 'ours', 'against', 'same', 'don', 'his', 'these', 'during', 'were', 'further', 'up', 'that', 'about', 'how', 'any', 'there', 'herself', 'was', 'down', 'did', 'himself', 'for', 'through', 'your', 'we', 'because', 'a', 'have', 'when', 'him', 'their', 'yourself', 'under', 'am', 'all', 'the', 'they', 'time', 'those', 'other', 'myself', 'being', 'ourselves', 'into', 'an', 'until', 'above', 'had', 's', 'only', 'where', 'yourselves', 'should', 'now', 'while', 'of', 'between', 'off', 'are', 'most', 'again', 'he', 'our', 'can', 'why', 'is', 'more', 'own', 'hers', 'has', 'to', 'and', 'it', 'once', 'whom', 'with', 'me', 'i', 'them', 'yours', 'will'}
    
    review = re.sub("[\.]", " ", review)
    review = re.sub("[\-]", " ", review)

    words = review.split(" ")

    feature_list = []

    t1 = str.maketrans('', '', digits)
    t2 = str.maketrans('', '', string.punctuation)

    for word in words:
        word = word.strip().lower()
        word = word.translate(t1)
        word = word.translate(t2)
        if word in stop_words:
            continue
        if word == '':
            continue
        feature_list.append(word)

    return feature_list


def count(data):
    priors = {"positive": 0, "negative": 0, "truthful": 0, "deceptive": 0}
    cond_prob = {"positive": {}, "negative": {},
                 "truthful": {}, "deceptive": {}}
    unique_words = []

    for review in data:
        labels = review[0].split("_")
        priors[labels[0]] += 1
        priors[labels[1]] += 1
        for word in review[1]:
            if word in cond_prob[labels[0]]:
                cond_prob[labels[0]][word] += 1
            else:
                cond_prob[labels[0]][word] = 1

            if word in cond_prob[labels[1]]:
                cond_prob[labels[1]][word] += 1
            else:
                cond_prob[labels[1]][word] = 1

            if word not in unique_words:
                unique_words.append(word)

    return priors, cond_prob, len(unique_words)


def calc_prob(priors, cond_prob, vocab_size, total_reviews):
    for p in priors:
        priors[p] = math.log(priors[p]/total_reviews)

    for label in cond_prob:
        total_words = len(cond_prob[label].keys())
        for word in cond_prob[label]:
            cond_prob[label][word] = math.log((
                cond_prob[label][word] + 1) / (total_words + vocab_size))


# Driver program
if __name__ == "__main__":
    train_data = []

    input_file_path = sys.argv[1]
    total_reviews = 0

    for root, dirs, files in os.walk(input_file_path):
        if not dirs:
            l = os.path.split(root)[0].split("/")
            label = l[-2].split("_")[0] + "_" + l[-1].split("_")[0]
            for file_name in files:
                file_path = root+"/"+file_name
                total_reviews += 1
                f = open(file_path, "r")
                review = f.read()
                cleaned_review = cleanup(review)
                train_data.append((label, cleaned_review))

    priors, cond_prob, vocab_size = count(train_data)
    calc_prob(priors, cond_prob, vocab_size, total_reviews)
    print_model(priors, cond_prob, vocab_size)
