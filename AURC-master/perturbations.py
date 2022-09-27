import json
import spacy
nlp = spacy.load("en_core_web_lg")
import re
import random
import secrets
from itertools import chain
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
from scipy import stats

with open("./AURC-master/data/AURC_DATA_dict.json","r") as f:
    AURC_DATA_dict = json.load(f)

def tokenize_spacy(sentence):
    doc = nlp(sentence)
    tokenized_sentence_spacy = [[token.text for token in s] for s in doc.sents]
    tokenized_sentence_spacy = list(chain.from_iterable(tokenized_sentence_spacy))  # flatten nested list if applicable
    return tokenized_sentence_spacy

def tokenize_bert(sentence):
    tokenized_sentence_bert = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence, add_special_tokens=False))
    return tokenized_sentence_bert

def create_dummy_entry(Train_OR_Dev_OR_Test):
    """
    Generate dummy entry to include in the robustness test sets (eval-only), otherwise error is risen.
    """
    assert Train_OR_Dev_OR_Test in ["Train","Dev","Test"]
    sentence = "--- This is a "+Train_OR_Dev_OR_Test+" dummy entry. ---"
    return {
        "Cross-Domain":
            Train_OR_Dev_OR_Test,
        "sentence":
            sentence,
        "sentence_level_stance":
            "non",
        "sentence_hash":
            "",
        "tokenized_sentence_bert":
            tokenize_bert(sentence),
        "tokenized_sentence_bert_labels":
            ("non "*len(tokenize_bert(sentence)))[:-1],
        "tokenized_sentence_spacy":
            tokenize_spacy(sentence),
        "tokenized_sentence_spacy_labels":
            ("non "*len(tokenize_spacy(sentence)))[:-1]
    }

########## PERTURBATIONS ########### ON CROSS-TOPIC MODEL
# We want to end up with two .json datasets per perturbation:
#   per1-before
#   per1-after
#   per2-before
#   per2-after
#   per3-before
#   per3-after
# Each dataset should be a list of dictionaries, where each dictionary has the following example format:
#{
#   "Cross-Domain": "Test",
#   "sentence": "Many pro-aborts privately think that abortion is often a good thing, but they rarely admit that is their real position.",
#   "sentence_hash": "c30dd94d8c83074c2505f37b8ef9474d", (i.e. before_after_pair_hash)
#   "sentence_level_stance": "pro",
#   "tokenized_sentence_bert": "Many pro - a ##bor ##ts privately think that abortion is often a good thing , but they rarely admit that is their real position .",
#   "tokenized_sentence_bert_labels": "non non non non non non non non non pro pro pro pro pro pro non non non non non non non non non non non",
#   "tokenized_sentence_spacy": "Many pro - aborts privately think that abortion is often a good thing , but they rarely admit that is their real position .",
#   "tokenized_sentence_spacy_labels": "non non non non non non non pro pro pro pro pro pro non non non non non non non non non non non"
#   },


#perturbation 1: ANN non-ARG segment + ARG segment <-VS-> ANN non-ARG segment + non-ARG segment, e.g. ’Some people may think that...’ + ’...it is not true.’

#code used to generate the candidates to annotate: no need to rerun this again
if False:
    with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per1/xxxxxANNnon-ARGsegments_candidates.tsv","w+") as f:
        f.write("{}\t{}\t{}\t{}\t{}\n".format("sentence_hash", "s2l", "segment","ANN","topic"))
        for topic, list_of_examples in AURC_DATA_dict.items():
            for x in list_of_examples:
                if x["Cross-Domain"] == "Test" and x["noisy_cleaned"]:
                    for segment,label in x["s2l"][:-1]: #ANN can per definition not be the last segment as it should precede a ARG segment
                        if label == "non":
                            f.write("{}\t{}\t{}\t{}\t{}\n".format(x["sentence_hash"],x["s2l"],segment,"",topic))

#initialize output dictionaries
AURC_DATA_dict_per1_before = {"gun control":[create_dummy_entry("Train"),create_dummy_entry("Dev")],
                              "school uniforms":[create_dummy_entry("Train"),create_dummy_entry("Dev")]}
AURC_DATA_dict_per1_after = {"gun control":[create_dummy_entry("Train"),create_dummy_entry("Dev")],
                             "school uniforms":[create_dummy_entry("Train"),create_dummy_entry("Dev")]}
#read from annotated tsv, and populate output dictionaries
with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per1/ANNnon-ARGsegments_candidates.tsv","r") as f:
    lines = f.readlines()
    for l in lines:
        if "\tx\t" in l: #if candidate examples has been annotated...
            print("lineL",l)
            _, \
            original_s2l, \
            ANNnonARG_segment, \
            __, \
            topic, \
            nonARG_segment_completion = l.split("\t")

            #remove trailing \n
            nonARG_segment_completion = nonARG_segment_completion.strip("\n")
            #lower-case first letter of the completing segment
            nonARG_segment_completion = nonARG_segment_completion[0].lower() + nonARG_segment_completion[1:]

            #transform tuples in string format to tuple format
            original_s2l = original_s2l.split("), (")
            ls = []
            for l in [s[-7:] for s in original_s2l]: #label is contained in the last 6 chars of the string
                if "non" in l:
                    ls.append("non")
                elif "pro" in l:
                    ls.append("pro")
                elif "con" in l:
                    ls.append("con")
            ss = [re.split(", \'(non|pro|con)",x)[0] for x in original_s2l]
            ss = [re.sub("(\'|\[|\]|\(|\)|\")","",x) for x in ss]
            ss = [x.strip() for x in ss]
            ss = [x[0].upper() + x[1:] for x in ss]
            assert len(ss) == len(ls) #if not, the label is not contained in the last 7 chars
            original_s2l = list(zip(ss,ls))

            #find the original, subsequent ARG segment that follows the ANNnonARG segment, to be used in the 'before' dataset
            ARG_segment_completion_index = -1
            for i,(segment,label) in enumerate(original_s2l):
                if i == ARG_segment_completion_index: #never true at first iter. Should read loop from the second if-clause onwards
                    ARG_segment_completion = segment
                    ARG_segment_completion = ARG_segment_completion[0].lower() + ARG_segment_completion[1:] #lower-case first letter
                    ARG_segment_completion_label = label+" "
                    break
                if segment == ANNnonARG_segment: #if True, our target is the segment in the subsequent iter!
                    ARG_segment_completion_index = i+1 #therefore, this will make sure that the first if-clause is True in the next iter

            #for the datasets we will need:
            #   ANNnonARG_segment +    ARG_segment_completion (-> before perturbation 1)
            #   ANNnonARG_segment + nonARG_segment_completion (-> after  perturbation 1)
            ANNnonARG_segment_tokenized_bert = " ".join(tokenize_bert(ANNnonARG_segment))
            ANNnonARG_segment_tokenized_spacy = " ".join(tokenize_spacy(ANNnonARG_segment))
            ANNnonARG_segment_tokenized_bert_labels = ("non "*len(ANNnonARG_segment_tokenized_bert.split()))[:-1]
            ANNnonARG_segment_tokenized_spacy_labels = ("non "*len(ANNnonARG_segment_tokenized_spacy.split()))[:-1]
            #
            ARG_segment_completion_tokenized_bert = " ".join(tokenize_bert(ARG_segment_completion))
            ARG_segment_completion_tokenized_spacy = " ".join(tokenize_spacy(ARG_segment_completion))
            ARG_segment_completion_tokenized_bert_labels = (ARG_segment_completion_label * len(ARG_segment_completion_tokenized_bert.split()))[:-1]
            ARG_segment_completion_tokenized_spacy_labels = (ARG_segment_completion_label * len(ARG_segment_completion_tokenized_spacy.split()))[:-1]
            #
            nonARG_segment_completion_tokenized_bert = " ".join(tokenize_bert(nonARG_segment_completion))
            nonARG_segment_completion_tokenized_spacy = " ".join(tokenize_spacy(nonARG_segment_completion))
            nonARG_segment_completion_tokenized_bert_labels = ("non "*len(nonARG_segment_completion_tokenized_bert.split()))[:-1]
            nonARG_segment_completion_tokenized_spacy_labels = ("non "*len(nonARG_segment_completion_tokenized_spacy.split()))[:-1]

            # unique hash for sentence pairs <before,after>
            sentence_hash = secrets.token_hex(nbytes=16)

            AURC_DATA_dict_per1_before[topic].append(
                {
                    "Cross-Domain":
                        "Test",
                    "sentence":
                        ANNnonARG_segment+" "+ARG_segment_completion,
                    "sentence_level_stance":
                        "pro", #psuedo-label (we only know it is ARG)
                    "sentence_hash":
                        sentence_hash,
                    "tokenized_sentence_bert":
                        ANNnonARG_segment_tokenized_bert+" "+ARG_segment_completion_tokenized_bert,
                    "tokenized_sentence_bert_labels":
                        ANNnonARG_segment_tokenized_bert_labels+" "+ARG_segment_completion_tokenized_bert_labels,
                    "tokenized_sentence_spacy":
                        ANNnonARG_segment_tokenized_spacy+" "+ARG_segment_completion_tokenized_spacy,
                    "tokenized_sentence_spacy_labels":
                        ANNnonARG_segment_tokenized_spacy_labels+" "+ARG_segment_completion_tokenized_spacy_labels
                }
            )
            AURC_DATA_dict_per1_after[topic].append(
                {
                    "Cross-Domain":
                        "Test",
                    "sentence":
                        ANNnonARG_segment+" "+nonARG_segment_completion,
                    "sentence_level_stance":
                        "non",
                    "sentence_hash":
                        sentence_hash,
                    "tokenized_sentence_bert":
                        ANNnonARG_segment_tokenized_bert+" "+nonARG_segment_completion_tokenized_bert,
                    "tokenized_sentence_bert_labels":
                        ANNnonARG_segment_tokenized_bert_labels+" "+nonARG_segment_completion_tokenized_bert_labels,
                    "tokenized_sentence_spacy":
                        ANNnonARG_segment_tokenized_spacy+" "+nonARG_segment_completion_tokenized_spacy,
                    "tokenized_sentence_spacy_labels":
                        ANNnonARG_segment_tokenized_spacy_labels+" "+nonARG_segment_completion_tokenized_spacy_labels
                }
            )
with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per1/AURC_DATA_dict_per1_before.json",'w') as f:
    json.dump(AURC_DATA_dict_per1_before, f, sort_keys=True, indent=4, separators=(',', ': '))
with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per1/AURC_DATA_dict_per1_after.json", 'w') as f:
    json.dump(AURC_DATA_dict_per1_after, f, sort_keys=True, indent=4, separators=(',', ': '))

#perturbation 2: ARG segment <-VS-> pure non-ARG sentence + ARG segment, e.g. : ’It’s a great service for parents as I was able to pick up lots of good stuff for little money ’ + ’uniforms force conformity.’.

#code used to generate the candidates to annotate: no need to rerun this again
if False:
    ARGsegments_candidates_guncontrol = []
    ARGsegments_candidates_schooluniforms = []
    for topic, list_of_examples in AURC_DATA_dict.items():
            for x in list_of_examples:
                if x["noisy_cleaned"] and x["Cross-Domain"] == "Test":
                    for segment,label in x["s2l"]:
                        if label in ["pro","con"]:
                            if topic == "gun control":
                                ARGsegments_candidates_guncontrol.append(segment)
                            elif topic == "school uniforms":
                                ARGsegments_candidates_schooluniforms.append(segment)
    nonARGsentences_candidates_guncontrol = []
    nonARGsentences_candidates_schooluniforms = []
    for topic, list_of_examples in AURC_DATA_dict.items():
        for x in list_of_examples:
            if x["noisy_cleaned"] == False and x["Cross-Domain"] == "Test":  # only pure non-ARG sentences
                for segment, label in x["s2l"]:
                    if label in ["non"]:
                        if topic == "gun control":
                            nonARGsentences_candidates_guncontrol.append(segment)
                        elif topic == "school uniforms":
                            nonARGsentences_candidates_schooluniforms.append(segment)

    random.shuffle(ARGsegments_candidates_guncontrol)
    random.shuffle(ARGsegments_candidates_schooluniforms)
    random.shuffle(nonARGsentences_candidates_guncontrol)
    random.shuffle(nonARGsentences_candidates_schooluniforms)

    cut_gc = min(len(ARGsegments_candidates_guncontrol),len(nonARGsentences_candidates_guncontrol))
    candidate_pairs_guncontrol = list(zip(ARGsegments_candidates_guncontrol[:cut_gc],nonARGsentences_candidates_guncontrol[:cut_gc]))
    cut_su = min(len(ARGsegments_candidates_schooluniforms),len(nonARGsentences_candidates_schooluniforms))
    candidate_pairs_schooluniforms = list(zip(ARGsegments_candidates_schooluniforms[:cut_su], nonARGsentences_candidates_schooluniforms[:cut_su]))

    with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per2/xxxxx_ARGsegments_non-ARGsentences_candidates.tsv","w") as f:
        f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format("ARGsegment","connector","nonARGsentence","annotation","length","topic"))
        for ARGsegment, nonARGsentence in candidate_pairs_guncontrol:
            length = str(len(ARGsegment.split()) + len("and besides ,".split()) + len(nonARGsentence.split()))
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(ARGsegment, "and besides,", nonARGsentence, "", length, "gun control"))
        for ARGsegment, nonARGsentence in candidate_pairs_schooluniforms:
            length = str(len(ARGsegment.split()) + len("and besides ,".split()) + len(nonARGsentence.split()))
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(ARGsegment, "and besides,", nonARGsentence, "", length, "school uniforms"))

#initialize output dictionaries
AURC_DATA_dict_per2_before = {"gun control":[create_dummy_entry("Train"),create_dummy_entry("Dev")],
                              "school uniforms":[create_dummy_entry("Train"),create_dummy_entry("Dev")]}
AURC_DATA_dict_per2_after = {"gun control":[create_dummy_entry("Train"),create_dummy_entry("Dev")],
                             "school uniforms":[create_dummy_entry("Train"),create_dummy_entry("Dev")]}
#forgot to store the labels of the ARGsegments; retrieving them (even though we will not use the pro-con granularity, just ARG-nonARG)
seg2lab = dict()
for topic, list_of_examples in AURC_DATA_dict.items():
    for x in list_of_examples:
        if x["noisy_cleaned"] and x["Cross-Domain"] == "Test":
            for segment,label in x["s2l"]:
                if label in ["pro","con"]:
                    seg2lab[segment] = label
#read from annotated tsv, and populate output dictionaries
with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per2/ARGsegments_non-ARGsentences_candidates.tsv","r") as f:
    lines = f.readlines()
    for l in lines:
        if "\tx\t" in l:  # if candidate examples has been annotated...
            ARGsegment, \
            connector, \
            nonARGsentence, \
            _, \
            __, \
            topic = l.split("\t")

            #remove trailing \n from topic
            topic = topic.strip("\n")
            # unique hash for sentence pairs <before,after>
            sentence_hash = secrets.token_hex(nbytes=16)
            # for some reason some ARGsegment have double quotes around them, and double quotes are doubled, so let's remove them
            ARGsegment = ARGsegment.strip('"')
            ARGsegment = re.sub('""','"',ARGsegment)
            # retrieve the forgotten label
            ARGsegment_label = seg2lab[ARGsegment]+" "

            # for the datasets we will need:
            #   ARG_segment                         (-> before perturbation 2)
            #   ARG_segment + nonARG_sentence       (-> after  perturbation 2)

            # Upper-case first letter of the ARG segment, and add full stop (which will be removed in the <after> dataset)
            ARGsegment = ARGsegment[0].upper() + ARGsegment[1:] + "."
            # Lower-case first letter of the nonARG sentence, concatenate connector with nonARG sentence, and add full stop
            nonARGsentence = connector + " " + nonARGsentence[0].lower() + nonARGsentence[1:] + "."

            ARGsegment_tokenized_bert = " ".join(tokenize_bert(ARGsegment))
            ARGsegment_tokenized_spacy = " ".join(tokenize_spacy(ARGsegment))
            ARGsegment_tokenized_bert_labels = (ARGsegment_label*len(ARGsegment_tokenized_bert.split()))[:-1]
            ARGsegment_tokenized_spacy_labels = (ARGsegment_label*len(ARGsegment_tokenized_spacy.split()))[:-1]

            AURC_DATA_dict_per2_before[topic].append(
                {
                    "Cross-Domain":
                        "Test",
                    "sentence":
                        ARGsegment,
                    "sentence_level_stance":
                        "pro",#pseudo-label (we only know it is ARG)
                    "sentence_hash":
                        sentence_hash,
                    "tokenized_sentence_bert":
                        ARGsegment_tokenized_bert,
                    "tokenized_sentence_bert_labels":
                        ARGsegment_tokenized_bert_labels,
                    "tokenized_sentence_spacy":
                        ARGsegment_tokenized_spacy,
                    "tokenized_sentence_spacy_labels":
                        ARGsegment_tokenized_spacy_labels
                }
            )
            # remove full stop, since it will now be concatenated with nonARGsentence
            ARGsegment = ARGsegment.strip(".")
            #
            ARGsegment_tokenized_bert = " ".join(tokenize_bert(ARGsegment))
            ARGsegment_tokenized_spacy = " ".join(tokenize_spacy(ARGsegment))
            ARGsegment_tokenized_bert_labels = (ARGsegment_label * len(ARGsegment_tokenized_bert.split()))[:-1]
            ARGsegment_tokenized_spacy_labels = (ARGsegment_label * len(ARGsegment_tokenized_spacy.split()))[:-1]
            #
            connector_tokenized_bert = " ".join(tokenize_bert(connector))
            connector_tokenized_spacy = " ".join(tokenize_spacy(connector))
            connector_tokenized_bert_labels = ("non " * len(connector_tokenized_bert.split()))[:-1]
            connector_tokenized_spacy_labels = ("non " * len(connector_tokenized_spacy.split()))[:-1]
            #
            nonARGsentence_tokenized_bert = " ".join(tokenize_bert(nonARGsentence))
            nonARGsentence_tokenized_spacy = " ".join(tokenize_spacy(nonARGsentence))
            nonARGsentence_tokenized_bert_labels = ("non " * len(nonARGsentence_tokenized_bert.split()))[:-1]
            nonARGsentence_tokenized_spacy_labels = ("non " * len(nonARGsentence_tokenized_spacy.split()))[:-1]

            AURC_DATA_dict_per2_after[topic].append(
                {
                    "Cross-Domain":
                        "Test",
                    "sentence":
                        ARGsegment+" "+connector+" "+nonARGsentence,
                    "sentence_level_stance":
                        "pro",#pseudo-label (we only know it is ARG)
                    "sentence_hash":
                        sentence_hash,
                    "tokenized_sentence_bert":
                        ARGsegment_tokenized_bert+" "+connector_tokenized_bert+" "+nonARGsentence_tokenized_bert,
                    "tokenized_sentence_bert_labels":
                        ARGsegment_tokenized_bert_labels+" "+connector_tokenized_bert_labels+" "+nonARGsentence_tokenized_bert_labels,
                    "tokenized_sentence_spacy":
                        ARGsegment_tokenized_spacy+" "+connector_tokenized_spacy+" "+nonARGsentence_tokenized_spacy,
                    "tokenized_sentence_spacy_labels":
                        ARGsegment_tokenized_spacy_labels+" "+connector_tokenized_spacy_labels+" "+nonARGsentence_tokenized_spacy_labels
                }
            )
with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per2/AURC_DATA_dict_per2_before.json",'w') as f:
    json.dump(AURC_DATA_dict_per2_before, f, sort_keys=True, indent=4, separators=(',', ': '))
with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per2/AURC_DATA_dict_per2_after.json", 'w') as f:
    json.dump(AURC_DATA_dict_per2_after, f, sort_keys=True, indent=4, separators=(',', ': '))

#perturbation 3: argumentative segments dataset from noisy examples

#code used to generate the candidates to annotate: no need to rerun this again
if False: #code used to generate the candidates to annotate: no need to rerun this again
    with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per3/xxxxx_ARGsegments_nonARGsegments_candidates.tsv","w+") as f:
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format("s2l",
                                                  "nonARGsegment_0",
                                                  "ARGsegment",
                                                  "nonARGsegment_1",
                                                  "ARGsegment_label",
                                                  "annotation",
                                                  "topic"))
        for topic, list_of_examples in AURC_DATA_dict.items():
            for x in list_of_examples:
                if x["noisy_cleaned"] and x["Cross-Domain"] == "Test":
                    if x["segment_labels_cleaned"] == ["non","pro"]:
                        (nonARGsegment_0,_),(ARGsegment,__) = x["s2l"]
                        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(x["s2l"],
                                                                      nonARGsegment_0,  #nonARG 0
                                                                      ARGsegment,       #ARG
                                                                      "empty",          #nonARG 1
                                                                      "pro",
                                                                      "",
                                                                      topic))
                    elif x["segment_labels_cleaned"] == ["non", "con"]:
                        (nonARGsegment_0, _), (ARGsegment, __) = x["s2l"]
                        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(x["s2l"],
                                                                      nonARGsegment_0,  #nonARG 0
                                                                      ARGsegment,       #ARG
                                                                      "empty",          #nonARG 1
                                                                      "con",
                                                                      "",
                                                                      topic))
                    elif x["segment_labels_cleaned"] == ["pro", "non"]:
                        (ARGsegment, __), (nonARGsegment_1, _) = x["s2l"]
                        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(x["s2l"],
                                                                      "empty",          #nonARG 0
                                                                      ARGsegment,       #ARG
                                                                      nonARGsegment_1,  #nonARG 1
                                                                      "pro",
                                                                      "",
                                                                      topic))
                    elif x["segment_labels_cleaned"] == ["con", "non"]:
                        (ARGsegment, __), (nonARGsegment_1, _)  = x["s2l"]
                        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(x["s2l"],
                                                                      "empty",          #nonARG 0
                                                                      ARGsegment,       #ARG
                                                                      nonARGsegment_1,  #nonARG 1
                                                                      "con",
                                                                      "",
                                                                      topic))

#initialize output dictionaries
AURC_DATA_dict_per3_before = {"gun control":[create_dummy_entry("Train"),create_dummy_entry("Dev")],
                              "school uniforms":[create_dummy_entry("Train"),create_dummy_entry("Dev")]}
AURC_DATA_dict_per3_after = {"gun control":[create_dummy_entry("Train"),create_dummy_entry("Dev")],
                             "school uniforms":[create_dummy_entry("Train"),create_dummy_entry("Dev")]}
#read from annotated tsv, and populate output dictionaries
with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per3/ARGsegments_nonARGsegments_candidates.tsv","r") as f:
    lines = f.readlines()
    for l in lines:
        if "\tx\t" in l:  # if candidate examples has been annotated...
            _, \
            nonARGsegment_0, \
            ARGsegment, \
            nonARGsegment_1, \
            ARGsegment_label, \
            __, \
            topic = l.split("\t")

            # remove trailing \n from topic
            topic = topic.strip("\n")
            # unique hash for sentence pairs <before,after>
            sentence_hash = secrets.token_hex(nbytes=16)
            # for some reason some ARGsegment have double quotes around them, and double quotes are doubled, so let's remove them
            ARGsegment = ARGsegment.strip('"')
            ARGsegment = re.sub('""', '"', ARGsegment)
            # add space to arg seg label, for later concatenation
            ARGsegment_label = ARGsegment_label+" "

            # for the datasets we will need:
            #   nonARG_segment + ARG_segment   OR    ARG_segment + nonARG_segment             (-> before perturbation 3)
            #   ARG_segment                                                                   (-> after  perturbation 3)

            if nonARGsegment_0 == "empty":
                ARGsegment = ARGsegment[0].upper() + ARGsegment[1:]
                nonARGsegment_1 = nonARGsegment_1[0].lower() + nonARGsegment_1[1:] + "."

                #
                ARGsegment_tokenized_bert = " ".join(tokenize_bert(ARGsegment))
                ARGsegment_tokenized_spacy = " ".join(tokenize_spacy(ARGsegment))
                ARGsegment_tokenized_bert_labels = (ARGsegment_label * len(ARGsegment_tokenized_bert.split()))[:-1]
                ARGsegment_tokenized_spacy_labels = (ARGsegment_label * len(ARGsegment_tokenized_spacy.split()))[:-1]
                #
                nonARGsegment_1_tokenized_bert = " ".join(tokenize_bert(nonARGsegment_1))
                nonARGsegment_1_tokenized_spacy = " ".join(tokenize_spacy(nonARGsegment_1))
                nonARGsegment_1_tokenized_bert_labels = ("non " * len(nonARGsegment_1_tokenized_bert.split()))[:-1]
                nonARGsegment_1_tokenized_spacy_labels = ("non " * len(nonARGsegment_1_tokenized_spacy.split()))[:-1]

                AURC_DATA_dict_per3_before[topic].append(
                    {
                        "Cross-Domain":
                            "Test",
                        "sentence":
                            ARGsegment+" "+nonARGsegment_1,
                        "sentence_level_stance":
                            "pro",#pseudo-label (we only know it is ARG)
                        "sentence_hash":
                            sentence_hash,
                        "tokenized_sentence_bert":
                            ARGsegment_tokenized_bert+" "+nonARGsegment_1_tokenized_bert,
                        "tokenized_sentence_bert_labels":
                            ARGsegment_tokenized_bert_labels+" "+nonARGsegment_1_tokenized_bert_labels,
                        "tokenized_sentence_spacy":
                            ARGsegment_tokenized_spacy+" "+nonARGsegment_1_tokenized_spacy,
                        "tokenized_sentence_spacy_labels":
                            ARGsegment_tokenized_spacy_labels+" "+nonARGsegment_1_tokenized_spacy_labels
                    }
                )

            elif nonARGsegment_1 == "empty":
                nonARGsegment_0 = nonARGsegment_0[0].upper() + nonARGsegment_0[1:]
                ARGsegment = ARGsegment[0].lower() + ARGsegment[1:] + "."

                #
                nonARGsegment_0_tokenized_bert = " ".join(tokenize_bert(nonARGsegment_0))
                nonARGsegment_0_tokenized_spacy = " ".join(tokenize_spacy(nonARGsegment_0))
                nonARGsegment_0_tokenized_bert_labels = ("non " * len(nonARGsegment_0_tokenized_bert.split()))[:-1]
                nonARGsegment_0_tokenized_spacy_labels = ("non " * len(nonARGsegment_0_tokenized_spacy.split()))[:-1]
                #
                ARGsegment_tokenized_bert = " ".join(tokenize_bert(ARGsegment))
                ARGsegment_tokenized_spacy = " ".join(tokenize_spacy(ARGsegment))
                ARGsegment_tokenized_bert_labels = (ARGsegment_label * len(ARGsegment_tokenized_bert.split()))[:-1]
                ARGsegment_tokenized_spacy_labels = (ARGsegment_label * len(ARGsegment_tokenized_spacy.split()))[:-1]

                AURC_DATA_dict_per3_before[topic].append(
                    {
                        "Cross-Domain":
                            "Test",
                        "sentence":
                            nonARGsegment_0+" "+ARGsegment,
                        "sentence_level_stance":
                            "pro",#pseudo-label (we only know it is ARG)
                        "sentence_hash":
                            sentence_hash,
                        "tokenized_sentence_bert":
                            nonARGsegment_0_tokenized_bert+" "+ARGsegment_tokenized_bert,
                        "tokenized_sentence_bert_labels":
                            nonARGsegment_0_tokenized_bert_labels+" "+ARGsegment_tokenized_bert_labels,
                        "tokenized_sentence_spacy":
                            nonARGsegment_0_tokenized_spacy+" "+ARGsegment_tokenized_spacy,
                        "tokenized_sentence_spacy_labels":
                            nonARGsegment_0_tokenized_spacy_labels+" "+ARGsegment_tokenized_spacy_labels,
                    }
                )

            # Upper-case first letter of the ARG segment, and add full stop
            ARGsegment = ARGsegment[0].upper() + ARGsegment[1:]
            if ARGsegment[-1] != ".":
                ARGsegment += "."

            ARGsegment_tokenized_bert = " ".join(tokenize_bert(ARGsegment))
            ARGsegment_tokenized_spacy = " ".join(tokenize_spacy(ARGsegment))
            ARGsegment_tokenized_bert_labels = (ARGsegment_label * len(ARGsegment_tokenized_bert.split()))[:-1]
            ARGsegment_tokenized_spacy_labels = (ARGsegment_label * len(ARGsegment_tokenized_spacy.split()))[:-1]

            AURC_DATA_dict_per3_after[topic].append(
                {
                    "Cross-Domain":
                        "Test",
                    "sentence":
                        ARGsegment,
                    "sentence_level_stance":
                        "pro",#pseudo-label (we only know it is ARG)
                    "sentence_hash":
                        sentence_hash,
                    "tokenized_sentence_bert":
                        ARGsegment_tokenized_bert,
                    "tokenized_sentence_bert_labels":
                        ARGsegment_tokenized_bert_labels,
                    "tokenized_sentence_spacy":
                        ARGsegment_tokenized_spacy,
                    "tokenized_sentence_spacy_labels":
                        ARGsegment_tokenized_spacy_labels
                }
            )
with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per3/AURC_DATA_dict_per3_before.json",'w') as f:
    json.dump(AURC_DATA_dict_per3_before, f, sort_keys=True, indent=4, separators=(',', ': '))
with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per3/AURC_DATA_dict_per3_after.json",'w') as f:
    json.dump(AURC_DATA_dict_per3_after, f, sort_keys=True, indent=4, separators=(',', ': '))

#CHECK COUNT
npg =0
ncg =0
nps =0
ncs =0
png =0
cng =0
pns =0
cns =0
with open("/Users/jonathankamp/PycharmProjects/repo-Trautmann/AURC-master/data/perturbations_dataset/per3/ARGsegments_nonARGsegments_candidates.tsv","r") as f:
    lines = f.readlines()
    for l in lines:
        if "\tx\t" in l:  # if candidate examples has been annotated...
            _, \
            nonARGsegment_0, \
            ARGsegment, \
            nonARGsegment_1, \
            ARGsegment_label, \
            __, \
            topic = l.split("\t")
            topic = topic.strip("\n")

            if topic == "gun control":
                if nonARGsegment_0 == "empty":
                    if ARGsegment_label == "pro":
                        png += 1
                    elif ARGsegment_label == "con":
                        cng += 1
                elif nonARGsegment_1 == "empty":
                    if ARGsegment_label == "pro":
                        npg += 1
                    elif ARGsegment_label == "con":
                        ncg += 1
            elif topic == "school uniforms":
                if nonARGsegment_0 == "empty":
                    if ARGsegment_label == "pro":
                        pns += 1
                    elif ARGsegment_label == "con":
                        cns += 1
                elif nonARGsegment_1 == "empty":
                    if ARGsegment_label == "pro":
                        nps += 1
                    elif ARGsegment_label == "con":
                        ncs += 1
print(npg,ncg,nps,ncs,png,cng,pns,cns)
