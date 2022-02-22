from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import dump_svmlight_file
from sklearn import metrics
import numpy as np
import logging
import codecs

import spacy
from collections import Counter
from spacy import displacy

logging.basicConfig(level=logging.INFO)

def parse_dataset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'rt', encoding="utf8") as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)

    return corpus, y


def featurize(corpus):
    '''
    Tokenizes and creates TF-IDF BoW vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: A sparse csr matrix of TFIDF-weigted ngram counts.
    '''

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    vectorizer = TfidfVectorizer(strip_accents="unicode", analyzer="word", tokenizer=tokenizer, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names()) # to manually check if the tokens are reasonable
    return X

# Exercise 1)
def Ex1(doc):
    word_frequencies = Counter()

    for tweet in doc:
        words = []
        for token in tweet: 
            # Let's filter out punctuation
            if not token.is_punct:
                words.append(token.text)
        word_frequencies.update(words)

    #Number of tokens:
    num_tokens = 0
    for tweet in doc:
        num_tokens = num_tokens + len(tweet)
    print("Number of tokens: ", num_tokens)

    #Number of types:
    num_types = len(word_frequencies.keys())
    print("Number of types: ", num_types)

    #Number of words:
    num_words = sum(word_frequencies.values())
    print("Number of words: ", num_words)

    #Average number of words per tweet:
    avg_words = 0
    num_words_temp = 0
    for tweet in doc:
        for token in tweet:
            if not token.is_punct:
                num_words_temp = num_words_temp + 1
    avg_words = num_words_temp/len(doc)
    print("Average number of words: ", avg_words)

    #Average word length:
    avg_word_len = 0
    ltemp = list(word_frequencies.keys())
    words_lens = np.array([len(word) for word in ltemp])
    words_frequencies = np.array(list(word_frequencies.values()))
    avg_word_len = (sum(words_lens*words_frequencies))/num_words
    print("Average length of word: ", (avg_word_len))

# Exercise 2
def Ex2(doc):
    word_frequencies = Counter()

    for tweet in doc:
        words = []
        for token in tweet: 
            # Let's filter out punctuation
            if not token.is_punct:
                words.append(token.text)
        word_frequencies.update(words)

    #Ten most frequent POS-tags
    tag_frequencies = Counter()
    word_frequencies = Counter()
    pos_frequencies = Counter()
    
    for tweet in doc:
        for sentence in tweet.sents:
            pos = []
            tag = []
            for token in sentence:
                pos.append(token.pos_)
                tag.append(token.tag_)
            pos_frequencies.update(pos)
            tag_frequencies.update(tag)

    top10 = [tag[0] for tag in tag_frequencies.most_common(10)]
    print("Top 10 Fined grained Tag frequencies: ", top10)

    #Universal POS-Tag: https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
    '''
    NN	NOUN
    NNP	PROPN
    IN	ADP
    .	PUNCT
    DT	DET
    PRP	PRON
    RB	ADV
    JJ	ADJ
    VB	VERB
    NNS	NOUN
    '''
    #Occurence:
    print()
    print("Top 10 Fined grained Tag number of occurences: " + str([tag[1] for tag in tag_frequencies.most_common(10)]))

    #Relative Tag Frequency(Not in % => Multiply results *100 to get %):
    total_tagged = sum(pos_frequencies.values())
    relative_frequencies = []
    for tag in tag_frequencies.most_common(10):
        relative_frequencies.append(tag[1]/total_tagged)
    print()
    print("Relative frequencies: ", relative_frequencies)
    print()

    #3 most frequent tokens with these tags:
    tags_word_freq = {str(tag) : [] for tag in top10}
    tags_word_freq2 = tags_word_freq.copy()
    for tweet in doc:
        for token in tweet:
            if token.tag_ in top10:
                tags_word_freq[token.tag_].append(token)

    for tag in tags_word_freq:
        tags_word_freq[tag] = set(tags_word_freq[tag])
        temp_freq = [str(elem) for elem in tags_word_freq[tag]]
        tags_word_freq2[tag] = [word for word in list(dict(sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True)).keys()) if word in temp_freq]

    for tag in tags_word_freq2:
        print(tag + ": ", tags_word_freq2[tag][:3])
    print()

    #Examples for an infrequent token with these tags:
    #for tag in tags_word_freq2:
    #    print(tag + ": ", tags_word_freq2[tag][-1])

# Exercise 4
def Ex4(doc):
    #Provide an example for a lemma that occurs in more than two inflections in the dataset.
    lemmas = {}
    for tweet in doc:
        for token in tweet:
            if str(token.lemma_) not in lemmas.keys():
                lemmas[str(token.lemma_)] = [str(token)]
            else:
                lemmas[str(token.lemma_)].append(str(token))

    for lemma in lemmas:
        lemmas[lemma] = set(lemmas[lemma])

    #print(lemmas)

    #Lemma: imagine
    #Inflected Forms: IMAGINE, Imagine, imagine, imagined
    #Example sentences for each form:
    example = {}
    for tweet in doc:
        for token in tweet:
            if token.lemma_ == 'imagine':
                example[token] = tweet
    print(example)

# Exercise 5
def Ex5(doc):
    #Number of named entities:
    print(sum(len(tweet.ents) for tweet in doc))

    #Number of different entity labels: 
    Nb_Elabels = set()
    for tweet in doc:
        for ent in tweet.ents:
            Nb_Elabels.add(ent.label_)
    print(len(Nb_Elabels))

    #Analyze the named entities in the first three tweets. Are they identified correctly? If not, explain your answer and propose a better decision.
    print()
    for tweet in doc[:3]:
        print(tweet)
        for ent in tweet.ents:
            print(ent.text, ent.label_)
            displacy.render(tweet, jupyter=True, style='ent')

if __name__ == "__main__":
    # Experiment settings

    # Dataset: SemEval2018-T4-train-taskA.txt or SemEval2018-T4-train-taskB.txt
    DATASET_FP = "./SemEval2018-T3-train-taskB.txt"
    TASK = "B" # Define, A or B
    FNAME = './predictions-task' + TASK + '.txt'
    PREDICTIONSFILE = open(FNAME, "w")

    K_FOLDS = 10 # 10-fold crossvalidation
    CLF = LinearSVC() # the default, non-parameter optimized linear-kernel SVM

    # Loading dataset and featurised simple Tfidf-BoW model
    corpus, y = parse_dataset(DATASET_FP)
    X = featurize(corpus)

    nlp = spacy.load('en_core_web_sm')

    # corpus = list of tweets
    doc = []
    for tweet in corpus:
        doc.append(nlp(tweet))
    
    print("Exercise 1")
    Ex1(doc)
    print("Exercise 2")
    Ex2(doc)
    print("Exercise 4")
    Ex4(doc)
    print("Exercise 5")
    Ex5(doc)