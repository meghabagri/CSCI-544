import sys
import json
import math
from collections import OrderedDict

# Driver program
if __name__ == "__main__":
    input_file_path = sys.argv[1]

    lines = []
    with open(input_file_path) as file:
        input_lines = file.readlines()
        for line in input_lines:
            lines.append(line.rstrip())
    
    all_tags = {}
    emission = {}
    transition = {}
    unique_word_tags = {}
    
    all_tags = {'START_TAG': len(lines)}
    transition['START_TAG'] = {}

    for line in lines:
        line = '/START_TAG' + " " + line
        line = line + " " + '/END_TAG'

        words = line.split(" ")
        prev_tag = 'START_TAG'

        for word in range(1, len(words)):
            # finding the rightmost '/' character
            word_tag = words[word].rpartition('/')
            word = word_tag[0]
            tag = word_tag[2]

            if tag != 'START_TAG' and tag != 'END_TAG':
                if tag in unique_word_tags.keys():
                    unique_word_tags[tag].add(word)
                else:
                    unique_word_tags[tag] = set((word))
            
            # counting tags given a tag
            if tag in transition[prev_tag]:
                transition[prev_tag][tag] += 1
            else:
                transition[prev_tag][tag] = 1

            if tag not in transition:
                transition[tag] = {} 
            prev_tag  = tag

            # count of all the tags
            if tag in all_tags:
                all_tags[tag] += 1     
            else:
                all_tags[tag] = 1

            if word == "":
                continue

            # counting words given a tag
            if word not in emission:
                emission[word] = {}
                emission[word][tag] = 1
            else:
                if tag in emission[word]:
                    emission[word][tag] += 1
                else:
                    emission[word][tag] = 1       
    
    # for laplace smoothing
    for prev_tag in all_tags.keys():
        for next_tag in all_tags.keys():
            if next_tag in transition[prev_tag]:
                transition[prev_tag][next_tag] += 1
            if next_tag not in transition[prev_tag]:
                transition[prev_tag][next_tag] = 1
            all_tags[next_tag] += 1

    # calculating probabilities
    for prev_tag in transition:
        for curr_tag in transition[prev_tag]:
            transition[prev_tag][curr_tag] = math.log(transition[prev_tag][curr_tag]) - math.log(all_tags[prev_tag])
                
    for word in emission:
        for tag in emission[word]:
            emission[word][tag] = math.log(emission[word][tag]) - math.log(all_tags[tag])
    
    for key, value in unique_word_tags.items():
        unique_word_tags[key] = len(list(value))
    unique_word_tags = OrderedDict(sorted(unique_word_tags.items(), key=lambda t: t[1]))
    unique_word_tags = list(unique_word_tags.keys())
    unique_word_tags = unique_word_tags[len(unique_word_tags)-5 : len(unique_word_tags)]

    with open('hmmmodel.txt', 'w') as file:
        file.write(json.dumps({"emission_prob": emission, "transmission_prob": transition, "unique_word_tags": unique_word_tags, "tag_counts": all_tags}, indent = 4))