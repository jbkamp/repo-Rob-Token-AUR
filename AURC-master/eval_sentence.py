# The files containing the sentence-wise predictions and true sentence-wise labels have the following format:
# NB: labels shown are assigned to each single token, but are actually just the sentence-wise label repeated n_tokens times
"""
{
    "0": {
        "sentence": "The lives saved by guns are no less precious , just because the media pay no attention to them .",
        "y_pred": "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "y_true": "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    },
    "1": {
        "sentence": "The fact that gun ownership encourages suicides is highly disturbing .",
        "y_pred": "1 1 1 1 1 1 1 1 1 1 1",
        "y_true": "2 2 2 2 2 2 2 2 2 2 2"
    },
    "2": {
        "sentence": "I know dont believe that it should be easier for criminals to get their hands on guns , but I know that states like California and New Jersey want more gun control than that .",
        "y_pred": "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "y_true": "2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
    },
    "3": {
        "sentence": "If we have gun control there will be fewer guns and consequently less crime .",
        "y_pred": "2 2 2 2 2 2 2 2 2 2 2 2 2 2 2",
        "y_true": "2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
    },
"""

import json
import sys
import numpy
# sys.argv[1] = <predictions_file_name>
# sys.argv[2] =
#   "model"             ...for token_f1, segment_f1, sentence_f1
#   "perturbations"     ...for binary precision,recall,f1 on perturbation datasets
# sys.argv[3] = <file_name_to_read_true_token_labels_from> (respective token-wise model must be run beforehand)
import re

def readpredictions(file_name, filename_to_read_true_token_labels_from):
    """
    reads json, returns dictionary

    Also: reads respective token-wise output to obtain true token-wise labels.
    Therefore, the respective token-wise model must be run beforehand
    """
    with open("./models/store_predictions_here/"+file_name, "r") as f:
        d = json.load(f)
    with open("./models/store_predictions_here/"+filename_to_read_true_token_labels_from, "r") as f2:
        d2 = json.load(f2)
    # add true token labels from d2 (tokenoutput) to d (sentenceoutput)
    for key, value in d2.items():
        true_token_labels = value["y_true"]
        d[key]["true_token_labels"] = true_token_labels
    return d

def f1_from_true_pred(list_true,list_pred):
    """
    list_true and list_pred should be lists of integers
    returns f1 score
    """
    classwise_f1scores = []
    for label in [0, 1, 2]:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for true, pred in list(zip(list_true, list_pred)):
            if true == label and pred == label:
                tp += 1
            elif true != label and pred != label:
                tn += 1
            elif true != label and pred == label:
                fp += 1
            elif true == label and pred != label:
                fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1score = 2 * (precision * recall) / (precision + recall)
        classwise_f1scores.append(f1score)

    f1 = sum(classwise_f1scores) / 3
    return f1

def compute_sentencef1(input_dict_of_predictions):
    """
    takes the dictionary of predictions
    returns sentence f1
    """
    all_sentences_true = []
    all_sentences_pred = []
    for key, value in input_dict_of_predictions.items():
        sentence_true = [int(p) for p in value["y_true"].split(" ")][0] #either index is fine since all tokens have assigned the true sentence label
        sentence_pred = [int(p) for p in value["y_pred"].split(" ")][0] #either index is fine since all tokens have assigned the pred sentence label
        all_sentences_true.append(sentence_true)
        all_sentences_pred.append(sentence_pred)
    sentence_f1 = f1_from_true_pred(all_sentences_true, all_sentences_pred)
    return sentence_f1

def compute_tokenf1(input_dict_of_predictions):
    """
    reads dictionary, returns token f1 score ( = avg over the three classes f1 scores)
    """
    all_tokens_true = []
    all_tokens_pred = []
    for key,value in input_dict_of_predictions.items():
        tokens_true = [int(p) for p in value["true_token_labels"].split(" ")]
        tokens_pred = [int(p) for p in value["y_pred"].split(" ")]
        all_tokens_true += tokens_true
        all_tokens_pred += tokens_pred
    token_f1 = f1_from_true_pred(all_tokens_true, all_tokens_pred)
    return token_f1

def tokens_2_segments(list_of_tokenwise_labels):
    """
    takes a list of token-wise labels
    returns a list of lists of token-wise labels, grouped by segment
    """
    list_of_tokenwise_labels.append(-999) # for the last case in the for-loop
    list_of_segment_lists = []
    current_segment = [list_of_tokenwise_labels[0]]
    for label in list_of_tokenwise_labels[1:]:
        if label == current_segment[-1]:
            current_segment.append(label)
        elif label != current_segment[-1]:
            list_of_segment_lists.append(current_segment)
            current_segment = [label]
    return list_of_segment_lists

def compute_segmentf1(input_dict_of_predictions):
    """
    reads dictionary, returns segment f1 score( = avg over per-sentence segment score)
    """
    sentence_segf1_scores = []
    for key,value in input_dict_of_predictions.items():
        tokens_true = [int(p) for p in value["true_token_labels"].split(" ")]
        tokens_pred = [int(p) for p in value["y_pred"].split(" ")]
        segments_true = tokens_2_segments(tokens_true)
        segments_pred = [] #should align the true segments
        cursor = 0
        for seg_true in segments_true:
            seg_pred = tokens_pred[cursor:cursor + len(seg_true)]
            segments_pred.append(seg_pred)
            cursor += len(seg_true)

        if len(segments_true) == 1 and len(segments_pred) == 1 and segments_true[0][0] == 0 and segments_pred[0][0] == 0:
            sentence_segf1 = 1
            sentence_segf1_scores.append(sentence_segf1)
        else:
            tp = 0
            #tn = 0
            fp = 0
            #fn = 0
            for i in range(len(segments_true)):
                intersected_tokens = 0
                true_seg_label = list(set(segments_true[i]))[0]
                for token_pred in segments_pred[i]:
                    if token_pred == true_seg_label:
                        intersected_tokens += 1
                r = intersected_tokens / len(segments_true[i])
                if r > 0.5:
                    if true_seg_label != 0:
                        tp += 1
                else:
                    fp += 1
            if (tp, fp) == (0, 0): # case in which there only are tns and/or fns, resulting in 0/0 below
                continue
            sentence_segf1 = tp / (tp + fp)
            sentence_segf1_scores.append(sentence_segf1)
    segment_f1 = sum(sentence_segf1_scores) / len(sentence_segf1_scores)
    return segment_f1

token_f1s = []
segment_f1s = []
sentence_f1s = []
fls = sys.argv[1].split(",")
for fl in fls:
    d = readpredictions(fl, sys.argv[3])     
    if sys.argv[2] == "model":
        print("_"*20)
        print("Eval file:\t\t\t\t",fl)
        print("Expected file with true token labels:\t", re.sub("_sentence","",fl))
        print("Selected file with true token labels:\t", sys.argv[3])
        print("-"*20)
        print("token_f1:\t{}".format(compute_tokenf1(d)))
        print("segment_f1:\t{}".format(compute_segmentf1(d)))
        print("sentence_f1:\t{}".format(compute_sentencef1(d)))
        print("_"*20)   
        token_f1s.append(compute_tokenf1(d))
        segment_f1s.append(compute_segmentf1(d))
        sentence_f1s.append(compute_sentencef1(d))
print("token|mean,std|\t",numpy.mean(token_f1s),numpy.std(token_f1s))
print("segment|mean,std|\t",numpy.mean(segment_f1s),numpy.std(segment_f1s))
print("sentence|mean,std|\t",numpy.mean(sentence_f1s),numpy.std(sentence_f1s))
print("_"*20)

########## eval perturbations: binary ARG vs non-ARG prec,rec,F1
def aprf1(input_dict_of_predictions):
    """
    input dictionary of predictions is transformed from multiclass to binary, where
        0 = 'non' (i.e. 0 stays 0)
        1 = 'pro'|'con' (i.e. 1 stays 1, and 2 becomes 1)
    returns accuracy, precision, recall, f1 score
    """
    all_sentences_true = []
    all_sentences_pred = []
    for key, value in input_dict_of_predictions.items():
        sentence_true = [int(p) for p in value["y_true"].split(" ")][0]  # either index is fine since all tokens have assigned the true sentence label
        sentence_pred = [int(p) for p in value["y_pred"].split(" ")][0]  # either index is fine since all tokens have assigned the pred sentence label
        if sentence_true == 2:
            sentence_true = 1
        if sentence_pred == 2:
            sentence_pred = 1
        all_sentences_true.append(sentence_true)
        all_sentences_pred.append(sentence_pred)

        per_example_predictions = [] # list of  0,1,0,0 etc. where 0 is correct prediction and 1 is incorrect ... to be used for subpopulations
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for true, pred in list(zip(all_sentences_true, all_sentences_pred)):
            if true == 1 and pred == 1:
                tp += 1
                per_example_predictions.append(1)
            elif true == 0 and pred == 0:
                tn += 1
                per_example_predictions.append(1)
            elif true == 0 and pred == 1:
                fp += 1
                per_example_predictions.append(0)
            elif true == 1 and pred == 0:
                fn += 1
                per_example_predictions.append(0)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if (tp + fp) == 0:
            precision = 0
            print("precision := 0, but as a result of 0 div!!")
        else:
            precision = tp / (tp + fp)
        
        if (tp + fn) == 0:
            recall = 0
            print("recall :=, but as a result of 0 div!!")
        else:
            recall = tp / (tp + fn)

        if (precision + recall) == 0:
            f1score = 0
            print("f1score := 0, but as a result of 0 div!!")
        else:
            f1score = 2 * (precision * recall) / (precision + recall)

    return (accuracy,precision,recall,f1score,per_example_predictions)

def meanstd(score_name, list_of_scores):
        return score_name+"|mean,std|\t"+str(numpy.mean(list_of_scores))+"\t"+str(numpy.std(list_of_scores))


if sys.argv[2] == "perturbations":
    accs = []
    precs = []
    recs = []
    f1scores = []
    fls = sys.argv[1].split(",")
    for fl in fls:
       d = readpredictions(fl, sys.argv[3])
       print("_"*20)
       print("PERTURBATIONS DATASET EVALUATION - SENTENCE-LEVEL")
       print("Eval file",fl)
       print("-"*20)
       a,p,r,f1 = aprf1(d)
       print("accuracy:\t{}".format(a))
       print("precision:\t{}".format(p))
       print("recall:\t{}".format(r))
       print("f1score:\t{}".format(f1))
       print("_"*20)
       accs.append(a)
       precs.append(p)
       recs.append(r)
       f1scores.append(f1)
    print(meanstd("acc",accs))
    print(meanstd("prec",precs))
    print(meanstd("rec",recs))
    print(meanstd("f1",f1scores))


if sys.argv[2] == "subpopulations":
    fls = sys.argv[1].split(",")
    for fl in fls:
        d = readpredictions(fl, sys.argv[3])
        _,__,___,per_example_predictions = aprf1(d)
        with open(fl+"_PEP.tsv","w") as f:
            f.write(str(per_example_predictions))
