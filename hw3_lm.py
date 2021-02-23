#!/usr/bin/env python
# coding: utf-8

# In[479]:


import sys
import numpy as np
import collections
import random


class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
    Parameters:
      n_gram (int): the n-gram order of the language model to create
      is_laplace_smoothing (bool): whether or not to use Laplace smoothing
    """
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing
        self.prob_matrix = None
        self.vocab_counts = None
        self.gram_counts = None
        self.gram_minus_counts = None

    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Parameters:
      training_file_path (str): the location of the training data to read

    Returns:
    None
    """
        training_data = open(training_file_path)
        content = training_data.read()
        training_list = content.split()

        for token in range(len(training_list)):
            if training_list.count(training_list[token].strip()) < 2:
                training_list[token] = self.UNK

        vocab = set(training_list)
        # sets the vocab_counts to the counts of each word in the vocabulary
        self.__setattr__('vocab_counts', collections.Counter(training_list))
        gram_counter = collections.Counter(self.create_n_grams(self.__getattribute__('n_gram'), training_list))
        gram_minus_counter = collections.Counter(self.create_n_grams(self.__getattribute__('n_gram') - 1,
                                                                     training_list)) \
 \
        # sets the gram_counts to the counts of each n_gram
        self.__setattr__('gram_counts', gram_counter)
        # sets the gram_minus_counts to the counts of each (n-1)-gram
        self.__setattr__('gram_minus_counts', gram_minus_counter)

        p_matrix = dict()
        # creates the probability matrix for the possible tokens and vocabulary
        for token in gram_minus_counter:
            p_matrix[token] = dict()
            for word in vocab:
                gram = token + ' ' + word
                num_appearances = gram_counter.setdefault(gram, 0)
                if self.__getattribute__('is_laplace_smoothing'):
                    prob = (num_appearances + 1) / (gram_minus_counter.get(token) + len(vocab))
                else:
                    prob = num_appearances / (gram_minus_counter.get(token))
                p_matrix[token][word] = prob

            self.__setattr__('prob_matrix', p_matrix)

    def create_n_grams(self, n, training_list):
        """Creates the n_grams with counts of each n_gram in the training set
        Parameters: n (int): is the number of tokens in the gram (tri-gram = 3)
        training_list (list): is the list of training data that has been split by whitespace"""
        gram_list = []
        # divide the data up into n-grams (unigrams, bigrams...)
        for current in range(len(training_list) - (n - 1)):
            gram = ''
            for i in range(current, current + n):
                gram = gram + " " + (training_list[i])
            gram_list.append(gram.strip())

        return gram_list

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
    Parameters:
      sentence (str): a sentence with tokens separated by whitespace to calculate the score of
      
    Returns:
      float: the probability value of the given string for this model
    """
        tokens = sentence.split()

        # makes all words that do not exist in the vocab into UNK tokens
        for token in range(len(tokens)):
            if tokens[token] not in self.__getattribute__('vocab_counts'):
                tokens[token] = self.UNK

        n_minus_tokens = self.create_n_grams(self.__getattribute__('n_gram') - 1, tokens)

        # remove the last n_minus token from the list so it does not check the prob of nothing after
        # the end sentence
        del n_minus_tokens[len(n_minus_tokens) - 1]

        # score begins at 1 because probability of the sentence start is 1
        score = 1
        word_num = self.__getattribute__('n_gram') - 1
        total_vocab_counts = sum(self.__getattribute__('vocab_counts').values())
        vocab_length = len(self.__getattribute__('vocab_counts'))
        if self.__getattribute__('n_gram') == 1:
            for token in tokens:
                tok_count = self.__getattribute__('vocab_counts').setdefault(token, 0)
                if self.__getattribute__('is_laplace_smoothing'):
                    score = score * ((tok_count + 1) / (total_vocab_counts + vocab_length))
                else:
                    score = score * (tok_count / total_vocab_counts)
        # Determine what the token is by splitting the sentence into n_grams first
        # and then going through the words one by one and getting the probability of n_gram + word
        else:
            for token in n_minus_tokens:
                gram_tok = token + ' ' + tokens[word_num]
                tok_count = self.__getattribute__('gram_counts').setdefault(gram_tok, 0)
                min_count = self.__getattribute__('gram_minus_counts').setdefault(token, 0)
                if self.__getattribute__('is_laplace_smoothing'):
                    new_score = (tok_count + 1) / (min_count + len(self.__getattribute__('vocab_counts')))
                else:
                    new_score = tok_count / min_count
                score = score * new_score
                word_num = word_num + 1

        return score

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      str: the generated sentence
    """
        current = self.SENT_BEGIN
        sentence = '<s>'

        # generates sentence for unigrams with randomly selecting words
        if self.__getattribute__('n_gram') == 1:
            vocab_list = list(self.__getattribute__('vocab_counts').keys())
            vocab_list.remove('<s>')
            while (current != self.SENT_END):
                current = vocab_list[random.randint(0, len(vocab_list) - 1)]
                sentence = sentence + " " + current
        else:

            # threshold probability based on what number of words in the vocabulary so above the threshold
            # means that it exists with greater probability than random
            threshold = 1 / len(self.__getattribute__('vocab_counts'))

            # filter out all of the words below the threshold probability
            p_matrix = (self.__getattribute__('prob_matrix')).copy()
            for row in p_matrix:
                delete = []
                for word, prob in p_matrix[row].items():
                    if prob < threshold:
                        delete.append(word)
                for w in delete:
                    del p_matrix[row][w]

            # for n-grams must randomly choose the 'first token' (the first one must contain <s>)
            # go through the n-gram list to find the ones with <s> tokens
            first_toks = self.__getattribute__('gram_counts').copy()
            delete = []
            for gram in first_toks:
                if self.SENT_BEGIN not in gram or first_toks[gram] == 0:
                    delete.append(gram)
            for g in delete:
                del first_toks[g]
            # generate the tokens for the sentence randomly for the filtered probability matrix with the first
            # from the filtered first_toks list

            first_list = list(first_toks.keys())
            first_list.remove('</s> <s>')
            current = first_list[random.randint(0, len(first_list) - 1)]
            curr = current.split()
            current = curr[1:]
            curr_string = ''
            for c in current:
                curr_string = curr_string + ' ' + c
            curr_string = curr_string.strip()
            sentence = sentence + ' ' + curr_string

            # creates the sentence based on the first token through random selection
            while (curr_string.strip() != self.SENT_END):
                word_list = list(p_matrix[curr_string.strip()].keys())
                curr_string = word_list[random.randint(0, len(word_list) - 1)]
                sentence = sentence + ' ' + curr_string

        return sentence

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
    Parameters:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing strings, one per generated sentence
    """
        sentence_list = []
        for i in range(n):
            current_sentence = self.generate_sentence()
            sentence_list.append(current_sentence)
        return sentence_list

    def print_test_info(self, test_path):
        """ Scores the testing file on the given test file and prints the name of the test file,
         number of sentences and average probability and standard deviation.
         Parameters:
             ""test_path (String): the file path for the test set"""
        testing_data = open(test_path)
        test = testing_data.read()
        test_split = test.split('\n')
        scores = []
        for sent in test_split:
            scores.append(self.score(sent))
        print('test corpus ' + test_path)
        print('# of sentences: ', len(test_split))
        print('Average probability ', np.mean(scores))
        print('Standard deviation ', np.std(scores))


def main():
    """ The given training file trains both of the models, unigram and bigram both with laplace,
    and both models are tested on the given testing files and the name of the test corpus, number
    of sentences, average probabilty and standard deviation and sentences are printed."""
    training_path = sys.argv[1]
    testing_path1 = sys.argv[2]
    testing_path2 = sys.argv[3]

    model_uni = LanguageModel(1, True)
    model_bi = LanguageModel(2, True)
    models = [model_uni, model_bi]
    for model in models:
        print('Model: ', model.__getattribute__('n_gram'), " Laplace smoothed")
        model.train(training_path)
        print('Sentences:')
        print(model.generate(50))
        model.print_test_info(testing_path1)
        model.print_test_info(testing_path2)


if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print("Usage:", "python hw3_lm.py training_file.txt testingfile1.txt testingfile2.txt")
        sys.exit(1)

    main()
