import spacy
from spacy.tokens import Doc
import pandas as pd
from spacy import displacy


def get_sentences(data):
    word_list = []
    for i in data['Word']:
        word_list.append(i)
    return word_list


def not_tokenize(tokens):
    return tokens


def get_doc(words):
    doc = Doc(nlp.vocab, words)
    # overtake the default tokenizer of the spacy parser.
    nlp.tokenizer = not_tokenize
    doc = nlp(doc)
    return doc


def get_dep_head(doc):
    head = []
    dep =[]
    for token in doc:
        dep.append(token.dep_)
        head.append(token.head)
    return dep, head


def get_head_number(head_list):
    head_numbers = []
    for head in head_list:
        i = head.i
        head_numbers.append(data['Number'][i])

    for i in range(len(head_numbers)):
        if head_numbers[i] == data['Number'][i]:
            head_numbers[i] = 0

    return head_numbers


# load data en specify nlp
nlp = spacy.load('en_core_web_sm')
data = pd.read_csv("conllst.2017.trial.simple.conll", delimiter="\t", names = ['Number', 'Word', 'Lemma', 'POS'])

# prep words for the use of the parser
word_list = get_sentences(data)
doc = get_doc(word_list)
dep_list, head_list = get_dep_head(doc)
head_numbers = get_head_number(head_list)
data['Head'] = head_numbers
data['Dependency label'] = dep_list

data.to_csv('CoNLL_Table.csv', index=False)
