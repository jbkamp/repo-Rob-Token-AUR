import json

with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/AURC_DATA_dict.json","r") as f:
    d = json.load(f)

def which_split_types():
    """
    Which split types are there? How are they divided per topic, and for the Cross-Domain selection and In-Domain selection?
    """
    for topic in d.keys():
        split_types_CD = []
        split_types_ID = []
        for entry in d[topic]:
            split_types_CD.append(entry['Cross-Domain'])
            split_types_ID.append(entry['In-Domain'])
        print("topic: [{}] --> unique_split_types_CD: [{}]".format(topic, set(split_types_CD)))
        print("topic: [{}] --> unique_split_types_ID: [{}]".format(topic, set(split_types_ID)))
        print()
def duplicates_exist(list_sentences):
    """
    Given a list, does it contain any duplicate elements?
    """
    if len(list_sentences) == len(set(list_sentences)):
        return "NO: duplicates not found"
    else:
        return "YES: duplicates found"
def is_data_leak(train,dev,test):
    """
    Given 3 input lists [train], [dev] and [test], are there any same elements that are contained in the (a,b) pairs where
        (a,b) == (train,dev)
        (a,b) == (train,test)
        (a,b) == (dev,test)
    ?
    If yes, at what index of list (a)?
    """
    leak_cache = 0
    for i,x in enumerate(train):
        if x in dev:
            print("YES: data leak TRAIN-DEV at index",i)
            leak_cache += 1
        if x in test:
            print("YES: data leak TRAIN-TEST at index",i)
            leak_cache += 1
    for i,x in enumerate(dev):
        if x in test:
            print("YES: data leak DEV-TEST at index",i)
            leak_cache += 1
    if leak_cache == 0:
        print("no leaks")
    else:
        print("n leaks:",leak_cache)
def print_this_nicely(header_string):
    print("---\n---\n---\n{}\n---".format(header_string))

print_this_nicely("Printing some example elements from the json dict")
print("d structure:")
for i,(k,v) in enumerate(list(d.items())):
    print("\t{}\tkey:{}\tvalue_type:{}\tvalue_len:{}\telems_in_value_type:{}".format(i,k,type(v),len(v),type(v[0])))

print("\nExamples of elems in value, for key=='abortion'") #value = list of elems
for i,elem in enumerate(d["abortion"][:5]): #elem = dictionary of (subkey:subvalue) pairs
    print(i)
    for subkey,subvalue in elem.items():
        print("\t{} : {}".format(subkey,subvalue))

print_this_nicely("Which split types are there? How are they divided per topic, and for the Cross-Domain selection and In-Domain selection?")
which_split_types()

print_this_nicely("In-Domain: Are there duplicates to be found within train, within dev, and within test?\nAre there data leaks to be found between train and dev, train and test, and dev and test?")
train_sents_CD = [] #initialising Cross-Domain sentence lists; these specific sentences are preselected by trautmann paper
dev_sents_CD = [] #initialising Cross-Domain sentence lists; these specific sentences are preselected by trautmann paper
test_sents_CD = [] #initialising Cross-Domain sentence lists; these specific sentences are preselected by trautmann paper

for topic in d.keys():
    train_sents_ID = [] #local variables, only for In-Domain stats, for each topic
    dev_sents_ID = [] #local variables, only for In-Domain stats, for each topic
    test_sents_ID = [] #local variables, only for In-Domain stats, for each topic
    for entry in d[topic]:
        tokenized_sentence_spacy = entry['tokenized_sentence_spacy'] #our 'sentences' are `tokenized_sentence_spacy`
        if entry['Cross-Domain'] == 'Train':
            train_sents_CD.append(tokenized_sentence_spacy)
        if entry['Cross-Domain'] == 'Dev':
            dev_sents_CD.append(tokenized_sentence_spacy)
        if entry['Cross-Domain'] == 'Test':
            test_sents_CD.append(tokenized_sentence_spacy)
        if entry['In-Domain'] == 'Train':
            train_sents_ID.append(tokenized_sentence_spacy)
        if entry['In-Domain'] == 'Dev':
            dev_sents_ID.append(tokenized_sentence_spacy)
        if entry['In-Domain'] == 'Test':
            test_sents_ID.append(tokenized_sentence_spacy)
    print("topic: [{}]".format(topic))
    print("sizes split train-dev-test: {}-{}-{}".format(len(train_sents_ID), len(dev_sents_ID), len(test_sents_ID)))
    print("train",duplicates_exist(train_sents_ID))
    print("dev",duplicates_exist(dev_sents_ID))
    print("test",duplicates_exist(test_sents_ID))
    is_data_leak(train_sents_ID, dev_sents_ID, test_sents_ID)
    print()

print_this_nicely("Cross-Domain: Are there duplicates to be found within train, within dev, and within test?\nAre there data leaks to be found between train and dev, train and test, and dev and test?")
print("sizes split train-dev-test: {}-{}-{}".format(len(train_sents_CD),len(dev_sents_CD),len(test_sents_CD)))
print("train",duplicates_exist(train_sents_CD))
print("dev",duplicates_exist(dev_sents_CD))
print("test",duplicates_exist(test_sents_CD))
is_data_leak(train_sents_CD,dev_sents_CD,test_sents_CD)
print()