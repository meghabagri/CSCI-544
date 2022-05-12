import numpy as np
from hmm import HMM


def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)

    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################

    for index, word in enumerate(unique_words.keys()):
        word2idx[word] = index

    for index, tag in enumerate(tags):
        tag2idx[tag] = index

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))

    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################

    # find initial state probabilities pi
    get_tags = [line.tags[0] for line in train_data]
    get_tags_count = [get_tags.count(key) for key in tag2idx.keys()]
    n = len(train_data)
    for s in range(S):
        pi[s] = get_tags_count[s] / n  # pi

    # find transition and emission probabilities
    for line in train_data:
        tag_id = [tag2idx[tag] for tag in line.tags]
        word_id = [word2idx[word] for word in line.words]
        for t in range(len(tag_id) - 1):
            A[tag_id[t], tag_id[t + 1]] += 1  # transition
            B[tag_id[t], word_id[t]] += 1  # emission
        B[tag_id[-1], word_id[-1]] += 1  # for the last index(T)

    A = np.divide(A, np.sum(A, axis=1))
    B = np.divide(B, np.sum(B, axis=1).reshape(len(B), 1))

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []

    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################

    trained_words_list = model.obs_dict.keys()

    test_words = []
    for line in test_data:
        test_words.extend(line.words)

    new_words = list(set(test_words) - set(trained_words_list))

    print("New Words Count: ", len(new_words))

    if new_words:
        n = len(trained_words_list)
        for i in new_words:
            model.obs_dict[i] = n
            n += 1

        temp_bso = np.divide(np.random.random((len(tags), len(new_words))), 100000000)

        model.B = np.append(model.B, temp_bso, axis=1)

    # predict tags using viterbi algorithm
    for line in test_data:
        tagging.append(model.viterbi(line.words))

    return tagging


# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
