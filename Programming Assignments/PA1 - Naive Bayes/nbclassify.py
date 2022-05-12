import sys
import os
import json
import string
from string import digits
import math
import re


def load_model():
    pathRead = 'nbmodel.txt'
    with open(pathRead, 'r') as openfile:
        json_object = json.load(openfile)

    return json_object['prior_prob'], json_object['cond_prob'], json_object['vocab_size']


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


def classify(review, prior_prob, cond_prob, vocab_size):
    label_PN = ""
    label_TD = ""
    class_prob = {"positive": prior_prob["positive"],
                  "negative": prior_prob["negative"],
                  "truthful": prior_prob["truthful"],
                  "deceptive": prior_prob["deceptive"]}

    for label in class_prob:
        for feature in review:
            if feature in cond_prob[label]:
                class_prob[label] += cond_prob[label][feature]
            else:
                p = len(cond_prob[label].keys()) + vocab_size
                class_prob[label] += math.log(1.0 / p)

    if class_prob["positive"] > class_prob["negative"]:
        label_PN = "positive"
    else:
        label_PN = "negative"

    if class_prob["truthful"] > class_prob["deceptive"]:
        label_TD = "truthful"
    else:
        label_TD = "deceptive"

    return label_PN, label_TD


# Driver program
if __name__ == "__main__":
    test_data = []
    input_file_path = sys.argv[1]

    for root, dirs, files in os.walk(input_file_path):
        if not dirs:
            for file_name in files:
                file_path = root+"/"+file_name

                f = open(file_path, "r")
                review = f.read()
                cleaned_review = cleanup(review)

                test_data.append(( file_path, cleaned_review))


    prior_prob, cond_prob, vocab_size = load_model()

    results = []
    for review in test_data:
        label_PN, label_TD = classify(
            review[1], prior_prob, cond_prob, vocab_size)
        results.append([label_TD, label_PN, review[0]])


    f = open("nboutput.txt", "w")
    for row in results:
        str1 = ' '.join(str(e) for e in row)
        f.write(str1 + "\n")

    f.close()
