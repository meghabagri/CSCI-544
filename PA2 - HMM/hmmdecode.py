import sys
import json

# Driver program
if __name__ == "__main__":
    # load the model
    model_file = open('hmmmodel.txt')
    model = json.load(model_file)

    # output file
    output = open("hmmoutput.txt", mode='w')

    # parameters
    emission_prob = model["emission_prob"]
    transmission_prob = model["transmission_prob"]
    tag_counts = model["tag_counts"]
    unique_word_tags = model["unique_word_tags"]

    input_file_path = sys.argv[1]
    lines = []
    with open(input_file_path) as file:
        input_lines = file.readlines()
        for line in input_lines:
            lines.append(line.rstrip())
    
    for line in lines:
        words = line.split(" ")
        probabilities = [{}]
        back_pointers = [{}]
        
        # probability for the first word
        word_exists = 0
        # check if a word is in the dictionary
        if words[0] in emission_prob.keys():
            word_exists = 1
            tags = emission_prob[words[0]].keys()
        else:
            tags = unique_word_tags
        for tag in tags:
            if word_exists:
                probabilities[0][tag] = transmission_prob['START_TAG'][tag] + emission_prob[words[0]][tag]
            else:
                probabilities[0][tag] = transmission_prob['START_TAG'][tag]
            back_pointers[0][tag] = 'START_TAG'

        # probabilities for the rest of the states
        for w in range(1, len(words)):
            word_exists = 0
            if words[w] in emission_prob.keys():
                word_exists = 1
                tags = emission_prob[words[w]].keys()
            else:
                tags = unique_word_tags
            
            probabilities.append({})
            back_pointers.append({})
            for tag in tags:
                if tag == 'START_TAG' or tag == "ENG_TAG":
                    continue
                max_prob = -sys.maxsize - 1
                max_state = ''
                for prev_tag in probabilities[w-1]:
                    if word_exists:
                        prob = transmission_prob[prev_tag][tag] + emission_prob[words[w]][tag] + probabilities[w-1][prev_tag]
                    else:
                        prob = transmission_prob[prev_tag][tag] + probabilities[w-1][prev_tag]
                    if max_prob < prob:
                        max_prob = prob
                        max_state = prev_tag
                probabilities[w][tag] = max_prob
                back_pointers[w][tag] = max_state

        # probability of the end word
        max_prob = -sys.maxsize - 1
        max_state = ''
        probabilities.append({})
        back_pointers.append({})
        for prev_tag in probabilities[len(words)-1]:
            prob = transmission_prob[prev_tag]['END_TAG'] + probabilities[len(words)-1][prev_tag]
            if max_prob < prob:
                max_prob = prob
                max_state = prev_tag
        probabilities[len(words)]['END_TAG'] = max_prob
        back_pointers[len(words)]['END_TAG'] = max_state

        l = len(probabilities)
        start = "END_TAG"
        taggings = words[l-2] + "/" + back_pointers[l-1][start]
        start = back_pointers[l-1][start]
        probabilities.pop()
        l -= 1

        while len(probabilities)-1:
            taggings = words[l-2] + "/" + back_pointers[l-1][start] + " "+ taggings
            start = back_pointers[l-1][start]
            l -= 1
            probabilities.pop()

        output.write(taggings)
        output.write('\n')
    output.close()
