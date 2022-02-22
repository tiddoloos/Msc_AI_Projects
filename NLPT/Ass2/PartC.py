import spacy
import pandas as pd
import csv


def print_different_numbers(gold, results):
    different_numbers = []
    for i in range(len(results['Number'])):
        if gold['Number'][i] != results['Number'][i]:
            different_numbers.append([gold['Token'][i], results['Token'][i]])
    if different_numbers == []:
        print('list is empty: no differences in numbering of the tokens!')
    else:
        print('these tokens differ in number:', different_numbers)


def find_root(data, ROOT):
    root = 0
    total_root = 0
    sentences = 0
    no_root_in_sent = 0
    sentences_without_root = []
    for i in range(len(data['Number'])):
        if data['Number'][i] == 1:
            if i > 0:
                if root == 0:
                    print('no root in this sentence')
                    no_root_in_sent += 1
                    sentences_without_root.append(data['Number'][i-1])
            sentences += 1
            root = 0
        if data['Dependency label'][i] == ROOT:
            root += 1
            total_root += 1
    if no_root_in_sent == 0 and sentences == total_root:
        print('all sentences have one root')
    else:
        print('these sentences ended without root', sentences_without_root)


def fill_differences(results):
    for i in range(len(results['Number'])):
        if results['Head'][i] == results['Gold head'][i]:
            results.loc[i, 'Dif head'] = 0
        if results['Dependency label'][i] == results['Gold dependency'][i]:
            results.loc[i, 'Dif dependency'] = 0
    return results


def errors_pos(results):
    pos_dict = {}
    for i in range(len(results['POS'])):
        pos = results['POS'][i]
        if pos not in pos_dict:
            count = {'occurence': 0, 'false_head' :0, 'false_dep': 0, }
            pos_dict[str(pos)]=count
        if results['Dif dependency'][i] == 1:
            pos_dict[str(pos)]['false_dep'] += 1
        if results['Dif head'][i] == 1:
            pos_dict[str(pos)]['false_head'] += 1
        pos_dict[str(pos)]['occurence'] += 1
    return pos_dict


def get_percentages(dict):
    for pos in dict:
        dict[pos]['head_error'] = round((dict[pos]['false_head']/dict[pos]['occurence'])*100, 2)
        dict[pos]['dep_error'] = round((dict[pos]['false_dep'] / dict[pos]['occurence'])*100, 2)
    return dict


def errors_dep(output):
    dep_dict = {}
    for i in range(len(output['Gold dependency'])):
        dep = output['Gold dependency'][i]
        if dep not in dep_dict:
            count = {'occurence': 0, 'false_dep': 0, 'dep_error': 0 }
            dep_dict[str(dep)] = count
        if output['Dif dependency'][i] == 1:
            dep_dict[str(dep)]['false_dep'] += 1
        dep_dict[str(dep)]['occurence'] += 1
    for dep in dep_dict:
        dep_dict[dep]['dep_error'] = round((dep_dict[dep]['false_dep'] / dep_dict[dep]['occurence'])*100, 2)
    return dep_dict


def get_frame(dict):
    frame = pd.DataFrame.from_dict(dict, orient='index')
    return frame


def Q8(gold, results):
    print('...checking for difference in numbers...')
    print_different_numbers(gold, results)
    print('...checking for root in sentences...')
    root_gold = 'root'
    root_results = 'ROOT'
    find_root(gold, root_gold)
    print('results data')
    find_root(results, root_results)

    print('...checking for differences...')
    output = fill_differences(results)
    return output


def Q9(output):
    pos_dict = errors_pos(output)
    pos_dict_perc = get_percentages(pos_dict)
    pos_errors = get_frame(pos_dict_perc)
    errors_by_head = pos_errors.sort_values(by=['head_error'], ascending=False)
    errors_by_dep = pos_errors.sort_values(by=['dep_error'], ascending=False)
    return errors_by_head, errors_by_dep


def Q10(output):
    dep_dict = errors_dep(output)
    dep_errors = get_frame(dep_dict)
    dep_errors = dep_errors.sort_values(by=['dep_error'], ascending=False)
    return dep_errors


# Setup data for partC
nlp = spacy.load('en_core_web_sm')
gold = pd.read_csv("conllst.2017.trial.simple.dep.conll", quoting=csv.QUOTE_NONE, delimiter="\t", names=['Number', 'Word', 'Lemma', 'POS', 'Head', 'Dependency label'])
results = pd.read_csv("CoNLL_Table.csv")
results['Gold head'] = gold['Head']
results['Gold dependency'] = gold['Dependency label']
results['Dif head'] = 1
results['Dif dependency'] = 1

# Q8
output = Q8(gold, results)
output.to_csv('Q8_differences.csv', index=False)

# Q9
# get data frames sorted on head and dependency
errors_by_head, errors_by_dep = Q9(output)
errors_by_head['head_error'].to_csv('Q9_errors_by_head.csv', index=True)
errors_by_dep['dep_error'].to_csv('Q9_errors_by_dep.csv', index=True)
# Q10
results_dep = Q10(output)
results_dep.to_csv('Q10_dependency_errors.csv', index=True)
