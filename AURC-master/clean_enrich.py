import json
import spacy
nlp = spacy.load("en_core_web_lg")
import time
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')

def get_segment_labels(token_labels):
    """
    token_labels is a list of token labels, e.g. [con,con,pro,con,non,non,non]
    Returns a list of segment labels, e.g.       [con,    pro,con,non        ]
    """
    assert type(token_labels) == list, "token_labels should be of type list"
    segment_labels = [token_labels[0]]  # initialized with the first of the token labels
    for l in token_labels[1:]:  # loop, excluding the already appended first label
        if l != segment_labels[-1]:  # if different from the last appended, append l (i.e. when l starts a new segment)
            segment_labels.append(l)
    return segment_labels

def convert_t2l_to_s2l(t2l):
    """
    input:
        t2l = list of tuples, e.g. [('Ourcustom', 'non'), ('made', 'non'), ('school', 'pro'), ('uniforms', 'pro'), ('are', 'pro')]
    output:
        s2l = list of tuples, e.g. [('Ourcustom made', 'non'), ('school uniforms are', 'pro')]
    """
    t2l.append(("XXX","LAST")) #add last pair so that the loop works
    segment = ""
    label = ""
    s2l = []
    for t,l in t2l:
        if l != label:                          #if new token has different label, means that previous segment is closed
            if segment != "":                   #if closed segment is not the empty initialized segment, append
                s2l.append((segment,label))
            segment = t                         #new segment is initialized with new token
            label = l                           #label becomes the label of the new segment
        else:                                   #if new token has same label as previous token, add token to segment
            segment = segment + " " + t
    return s2l

def retrieve_train_dev_test_sentences(dataset_dict,domain):
    train_sentences = set()
    dev_sentences = set()
    test_sentences = set()
    for topic, list_of_examples in dataset_dict.items():
        for x in list_of_examples:
            if x[domain] == "Train":
                train_sentences.add(x["sentence"])
            elif x[domain] == "Dev":
                dev_sentences.add(x["sentence"])
            elif x[domain] == "Test":
                test_sentences.add(x["sentence"])
    return train_sentences,dev_sentences,test_sentences

def clean_remove_duplicates(dataset_dict):
    """
    Steps (1) and (2), where (2) is applied on the output of (1).
    1) removing duplicates between train and dev, train and test, and dev and test:
        train element will be removed in favor of dev|test element
        dev element will be removed in favor of test element
    2) removing duplicates within train, dev, and test:
        unique instances will be the result of this operation
    """
    doms = ["In-Domain", "Cross-Domain"]
    # STEP (1)
    for dom in doms:
        removed_train_dev = 0
        removed_train_test = 0
        removed_dev_test = 0
        _,dev_sentences,test_sentences = retrieve_train_dev_test_sentences(dataset_dict,dom)
        for topic,list_of_examples in dataset_dict.items():
            for x in list_of_examples:
                if x[dom] == "Train" and x["sentence"] in dev_sentences:
                    x[dom] = "nan"
                    removed_train_dev += 1
                elif x[dom] == "Train" and x["sentence"] in test_sentences:
                    x[dom] = "nan"
                    removed_train_test += 1
                elif x[dom] == "Dev" and x["sentence"] in test_sentences:
                    x[dom] = "nan"
                    removed_dev_test += 1
        print(dom, removed_train_dev,removed_train_test,removed_dev_test) # ID: 2-5-1; CD: 0-0-0
    # STEP (2)
    for dom in doms:
        train_removed = 0
        dev_removed = 0
        test_removed = 0
        for topic,list_of_examples in dataset_dict.items():
            for i,x in enumerate(list_of_examples):
                for y in list_of_examples[:i]+list_of_examples[i+1:]:
                    if x[dom] == "Train" and y[dom] == "Train" and x["sentence"] == y["sentence"]:
                        x[dom] == "nan"
                        train_removed += 1
                    elif x[dom] == "Dev" and y[dom] == "Dev" and x["sentence"] == y["sentence"]:
                        x[dom] == "nan"
                        dev_removed += 1
                    elif x[dom] == "Test" and y[dom] == "Test" and x["sentence"] == y["sentence"]:
                        x[dom] == "nan"
                        test_removed += 1
        print(dom, train_removed, dev_removed, test_removed)
    return dataset_dict

def enrich(dataset_dict):
    """
    add entries to input dataset_dict about token labels, segment labels, sentence labels (noisy , noisy_cleaned)
    """
    for topic, list_of_examples in dataset_dict.items():
        for x in list_of_examples:
            token_labels = x["tokenized_sentence_spacy_labels"].split(" ")
            x["segment_labels"] = get_segment_labels(token_labels)
            if len(x["segment_labels"]) > 1:
                x["noisy"] = True
            else:
                x["noisy"] = False
            x["token2labelmapping_original"] = list(zip(x["tokenized_sentence_spacy"].split(" "), x["tokenized_sentence_spacy_labels"].split(" ")))
            punctuation = [".", "..", "...", "....", ".....", "?", "!", ")", "(", "/", "\\", "”", "'", '"']
            x["token2labelmapping_which_NONpunctremoved"] = [(token, label) for (token, label) in x["token2labelmapping_original"] if token in punctuation and label == "non"]
            x["token2labelmapping_cleaned"] = [(token, label) for (token, label) in x["token2labelmapping_original"] if not (token in punctuation and label == "non")]
            x["tokenized_sentence_spacy_cleaned"] = " ".join([token for (token, label) in x["token2labelmapping_cleaned"]])
            x["tokenized_sentence_spacy_labels_cleaned"] = " ".join([label for (token, label) in x["token2labelmapping_cleaned"]])
            x["segment_labels_cleaned"] = get_segment_labels(x["tokenized_sentence_spacy_labels_cleaned"].split(" "))
            if len(x["segment_labels_cleaned"]) > 1:
                x["noisy_cleaned"] = True
            else:
                x["noisy_cleaned"] = False
            x["s2l"] = convert_t2l_to_s2l(x["token2labelmapping_cleaned"])
    return dataset_dict

def compute_arg_ratio(t2l): #does not differentiate among PRO and CON
    """
    Computes the ratio of argumentative tokens over the total tokens in the sentence.
    Takes t2l, which is a list of (token,label) tuples.
    Returns ratio (float).

    e.g.
        compute_arg_ratio([("1","con"),("2","con"),("3","pro"),("4","non"),("5","non"),("6","non"),("7","non"),("8","non"),("9","non"),("10","pro")])
    -> 0.4
    """
    n_tokens_sentence = len(t2l)
    n_tokens_arg = len([(t,l) for (t,l) in t2l if l != "non"])
    ratio = n_tokens_arg / n_tokens_sentence
    return ratio

def enrich_maxsim_and_ratio_scores(dataset_dict):
    """
    Takes dataset_dict
    Computes, for each noisy test sentence in dataset_dict, a similarity score in relation to each sentence in list_of_sentences.
    The max of those scores is kept.
    """
    training_sentences = []
    for topic, list_of_examples in dataset_dict.items():
        for x in list_of_examples:
            if x["Cross-Domain"] == "Train":
                training_sentences.append((x["tokenized_sentence_spacy"],x["sentence_level_stance"]))

    for topic, list_of_examples in dataset_dict.items():
        if topic in ["gun control","school uniforms"]:
            print(topic)
            for i,x in enumerate(list_of_examples):
                start = time.time()
                if x["Cross-Domain"] == "Test" and x["noisy"]:
                    test_s = x["tokenized_sentence_spacy"]
                    sim_score = -1
                    sim_score_nonARG = -1
                    for train_s, train_s_stance in training_sentences:
                        temp_sim_score = nlp(test_s).similarity(nlp(train_s))
                        if temp_sim_score > sim_score:
                            sim_score = temp_sim_score
                        if train_s_stance == "non":
                            if temp_sim_score > sim_score_nonARG:
                                sim_score_nonARG = temp_sim_score
                    x["max_simscore"] = sim_score
                    x["max_simscore_nonARG"] = sim_score_nonARG
                    x["arg_ratio"] = compute_arg_ratio(x["token2labelmapping_cleaned"])
                tm = time.time()-start
                print(i+1, "done out of", len(list_of_examples), tm, "sec")
    return dataset_dict

if False:
    with open("./AURC-master/data/backup_07032022/AURC_DATA_dict.json","r") as infile:
        AURC_DATA_dict = json.load(infile)

    AURC_DATA_dict = clean_remove_duplicates(AURC_DATA_dict)
    AURC_DATA_dict = enrich(AURC_DATA_dict)
    # !! TAKES 2 DAYS !!# !! TAKES 2 DAYS !! # !! TAKES 2 DAYS !! # !! TAKES 2 DAYS !! # !! TAKES 2 DAYS !!
    AURC_DATA_dict = enrich_maxsim_and_ratio_scores(AURC_DATA_dict) # !! TAKES 2 DAYS !! # !! TAKES 2 DAYS !!

if False:
    #save as "AURC_DATA_cleaned_please_rename.json" temporarily; manually renaming to AURC_DATA_dict.json, whereas the original will be moved to /backup_07032022
    with open("./AURC-master/data/AURC_DATA_dict_cleaned_please_rename.json","w") as outfile:
        json.dump(AURC_DATA_dict, outfile, sort_keys=True, indent=4, separators=(',', ': '))

#stats dataset to report in paper
with open("./AURC-master/data/AURC_DATA_dict.json","r") as f:
    AURC_DATA_dict = json.load(f)
for topic,list_of_examples in AURC_DATA_dict.items():
    print(topic)
    in_train = 0
    in_dev = 0
    in_test = 0
    cr_train = 0
    cr_dev = 0
    cr_test = 0
    for x in list_of_examples:
        ind = x["In-Domain"]
        if ind == "Train":
            in_train += 1
        elif ind == "Dev":
            in_dev += 1
        elif ind == "Test":
            in_test += 1
        crd = x["Cross-Domain"]
        if crd == "Train":
            cr_train += 1
        elif crd == "Dev":
            cr_dev += 1
        elif crd == "Test":
            cr_test += 1
    print("in",in_train,in_dev,in_test)
    print("cr",cr_train,cr_dev,cr_test)

n_arg_sents = 0
n_nonarg_sents = 0
fully_pro_sents = 0
fully_con_sents = 0
mixed = 0
for topic,list_of_examples in AURC_DATA_dict.items():
    for x in list_of_examples:
        if x['sentence_level_stance'] in ["pro","con"]:
            n_arg_sents +=1
            if x["segment_labels_cleaned"] == ["pro"]:
                fully_pro_sents += 1
            elif x["segment_labels_cleaned"] == ["con"]:
                fully_con_sents += 1
            else:
                mixed += 1

        else:
            n_nonarg_sents +=1

print("n_arg_sents",n_arg_sents)
print("n_nonarg_sents",n_nonarg_sents)
print("fully_pro_sents",fully_pro_sents)
print("fully_con_sents",fully_con_sents)
print("mixed",mixed)