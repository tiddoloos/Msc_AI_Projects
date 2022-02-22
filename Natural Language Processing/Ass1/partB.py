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
from random import randint
from random import seed

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

# Exercise 1
def Ex1(doc,y):
    #Analyze the number of instances for each of the four classification labels.
    #Class Label | Instances | Relative Label Frequency(%) | Example sentence with this label

    labels = [0,1,2,3]
    for i in labels:
        print("Class label " + str(i))
        print("Instances: " + str(y.count(i)))
        print("Relative label Frequency(%): " + str((y.count(i)/len(y))*100))
        tweet = 0
        while y[tweet] != i:
            tweet = tweet + 1
        print("Example sentence with this label: " + str(doc[tweet]))

# Exercise 2 - Utility functions
def multiclass_acc (matrix,class_label):
    TP = matrix[class_label][class_label]
    TN = sum(matrix[i][j] for i in range(4) for j in range(4) if i!=class_label and j!=class_label)
    TOTAL = matrix.sum()
    return (TP+TN)/TOTAL

def acc_weighted (accuracies,matrix,labels):
    acc = []
    for i in range(len(labels)):
        acc.append(accuracies[i]*matrix[i][i])
    acc = sum(acc)/matrix.sum()
    return acc

def Ex2(doc,y):
    #Calculate two baselines and evaluate their performance on the test set.
    #Class | Accuracy | Precision | Recall | F1 | macro-average | weighted average
    # metrics.accuracy_score | metrics.precision_score | metrics.recall_score | metrics.f1_score
    # https://scikit-learn.org/stable/modules/model_evaluation.html

    seed(5)
    labels = [0,1,2,3]

    #Random baseline (randomly assigns one of the four classification labels. Make sure to fix the random seed and average the results over 100 iterations)
    print("RANDOM BASELINE")

    random_final_results = []
    random_acc = []
    random_prec = []
    random_recall = []
    random_f1 = []

    random_acc_macro = []
    random_acc_weighted = []
    random_prec_macro = []
    random_prec_weighted = []
    random_recall_macro = []
    random_recall_weighted = []
    random_f1_macro = []
    random_f1_weighted = []

    #Make iterations
    for i in range(100):
        random_acc_temp = []
        random_final_results = [randint(0,3) for i in range(len(y))] #have final results
        
        #make metrics
        matrix = metrics.confusion_matrix(y,random_final_results,labels=[0,1,2,3]) #beginning of accuracy per class
        for i in range(4):
            random_acc_temp.append(multiclass_acc(matrix,i))
        random_acc.append(random_acc_temp)
        random_prec.append(metrics.precision_score(y,random_final_results,average=None))
        random_recall.append(metrics.recall_score(y,random_final_results,average=None))
        random_f1.append(metrics.f1_score(y,random_final_results,average=None))
        
        random_acc_macro.append(metrics.accuracy_score(y,random_final_results))
        random_acc_weighted.append(acc_weighted(random_acc_temp,matrix,labels))
        random_prec_macro.append(metrics.precision_score(y,random_final_results,average='macro'))
        random_prec_weighted.append(metrics.precision_score(y,random_final_results,average='weighted'))
        random_recall_macro.append(metrics.recall_score(y,random_final_results,average='macro'))
        random_recall_weighted.append(metrics.recall_score(y,random_final_results,average='weighted'))
        random_f1_macro.append(metrics.f1_score(y,random_final_results,average='macro'))
        random_f1_weighted.append(metrics.f1_score(y,random_final_results,average='weighted'))
        
    #average 100 metrics
    random_acc = [sum(value[0] for value in random_acc)/100,sum(value[1] for value in random_acc)/100,sum(value[2] for value in random_acc)/100,sum(value[3] for value in random_acc)/100]
    print(random_acc)
    random_prec = [sum(value[0] for value in random_prec)/100,sum(value[1] for value in random_prec)/100,sum(value[2] for value in random_prec)/100,sum(value[3] for value in random_prec)/100]
    print(random_prec)
    random_recall = [sum(value[0] for value in random_recall)/100,sum(value[1] for value in random_recall)/100,sum(value[2] for value in random_recall)/100,sum(value[3] for value in random_recall)/100]
    print(random_recall)
    random_f1 = [sum(value[0] for value in random_f1)/100,sum(value[1] for value in random_f1)/100,sum(value[2] for value in random_f1)/100,sum(value[3] for value in random_f1)/100]
    print(random_f1)

    print()
    print(sum(random_acc_macro)/100)
    print(sum(random_acc_weighted)/100)
    print(sum(random_prec_macro)/100)
    print(sum(random_prec_weighted)/100)
    print(sum(random_recall_macro)/100)
    print(sum(random_recall_weighted)/100)
    print(sum(random_f1_macro)/100)
    print(sum(random_f1_weighted)/100)

    #Majority baseline (always assigns the majority class: aka label 0)
    print()
    print("MAJORITY BASELINE")
    majority_final_results = [0 for i in range(len(y))]
    matrix = metrics.confusion_matrix(y,majority_final_results,labels=[0,1,2,3]) #beginning of accuracy per class
    majority_acc = []
    for i in range(4):
        majority_acc.append(multiclass_acc(matrix,i))
    majority_prec = metrics.precision_score(y,majority_final_results,average=None) #ignore warning: add parameter [labels=[0]] to avoid it
    majority_recall = metrics.recall_score(y,majority_final_results,average=None)
    majority_f1 = metrics.f1_score(y,majority_final_results,average=None)

    majority_acc_macro = metrics.accuracy_score(y,majority_final_results)
    majority_acc_weighted = acc_weighted(majority_acc,matrix,labels)
    majority_prec_macro = metrics.precision_score(y,majority_final_results,average='macro',labels=[0])
    majority_prec_weighted = metrics.precision_score(y,majority_final_results,average='weighted',labels=[0])
    majority_recall_macro = metrics.recall_score(y,majority_final_results,average='macro')
    majority_recall_weighted = metrics.recall_score(y,majority_final_results,average='weighted')
    majority_f1_macro = metrics.f1_score(y,majority_final_results,average='macro')
    majority_f1_weighted = metrics.f1_score(y,majority_final_results,average='weighted')

    print(majority_acc)
    print(majority_prec)
    print(majority_recall)
    print(majority_f1)

    print()
    print(majority_acc_macro)
    print(majority_acc_weighted)
    print(majority_prec_macro)
    print(majority_prec_weighted)
    print(majority_recall_macro)
    print(majority_recall_weighted)
    print(majority_f1_macro)
    print(majority_f1_weighted)

if __name__ == "__main__":
    # Experiment settings

    # Dataset: SemEval2018-T4-train-taskA.txt or SemEval2018-T4-train-taskB.txt
    DATASET_FP = "./SemEval2018-T3_gold_test_taskB_emoji.txt"
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
    Ex1(doc,y)
    print("Exercise 2")
    Ex2(doc,y)