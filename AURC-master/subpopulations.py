import json
from scipy import stats
import numpy

with open("./data/AURC_DATA_dict.json","r") as f:
    AURC_DATA_dict = json.load(f)

maxsim1 = [] 
maxsim2 = []
argratio = []

for topic in ["gun control","school uniforms"]:
    for x in AURC_DATA_dict[topic]:
        if x["Cross-Domain"] == "Test":
            if "max_simscore" in x.keys():
                if x["max_simscore"] > x["max_simscore_nonARG"]:
                    maxsim1.append(x["max_simscore"])
                    maxsim2.append(-999)
                else:
                    maxsim1.append(-999)
                    maxsim2.append(x["max_simscore_nonARG"])
                argratio.append(x["arg_ratio"])
            else:
                argratio.append(-999)

PEP_files = [
        "aurc_cr_predictions_test_001.json_PEP.tsv",
        "aurc_cr_predictions_test_002.json_PEP.tsv",
        "aurc_cr_predictions_test_003.json_PEP.tsv",
        "aurc_cr_predictions_test_004.json_PEP.tsv",
        "aurc_cr_predictions_test_005.json_PEP.tsv",
        "aurc_cr_crf_predictions_test_001.json_PEP.tsv",
        "aurc_cr_crf_predictions_test_002.json_PEP.tsv",
        "aurc_cr_crf_predictions_test_003.json_PEP.tsv",
        "aurc_cr_crf_predictions_test_004.json_PEP.tsv",
        "aurc_cr_crf_predictions_test_005.json_PEP.tsv",
        "aurc_cr_predictions_test_sentence_001.json_PEP.tsv",
        "aurc_cr_predictions_test_sentence_002.json_PEP.tsv",
        "aurc_cr_predictions_test_sentence_003.json_PEP.tsv",
        "aurc_cr_predictions_test_sentence_004.json_PEP.tsv",
        "aurc_cr_predictions_test_sentence_005.json_PEP.tsv"
        ]

PEPs = dict()
for pred_file in PEP_files:
    with open("./models/store_predictions_here/"+pred_file,"r") as f:
        list_predictions = f.readlines()[0]
        list_predictions = list_predictions.strip("[")
        list_predictions = list_predictions.strip("]")
        list_predictions = list_predictions.split(",")
        list_predictions = [int(p) for p in list_predictions]
        PEPs[pred_file] = list_predictions

print("Total length noisy+not noisy test set Cross-Domain")
print(len(list_predictions))

for filename,list_predictions in PEPs.items():
    PEPs[filename] = {
            "maxsim1":[v for (p,v) in [(pred,value) for (pred,value) in list(zip(list_predictions,maxsim1)) if value != -999]],
            "maxsim2":[v for (p,v) in [(pred,value) for (pred,value) in list(zip(list_predictions,maxsim2)) if value != -999]],
            "argratio":[v for (p,v) in [(pred,value) for (pred,value) in list(zip(list_predictions,argratio)) if value != -999]],
            "list_predictions_noisy_maxsim1": [p for (p, v) in [(pred, value) for (pred, value) in list(zip(list_predictions,maxsim1)) if value != -999]],
            "list_predictions_noisy_maxsim2": [p for (p, v) in [(pred, value) for (pred, value) in list(zip(list_predictions,maxsim2)) if value != -999]],
            "list_predictions_noisy_argratio":[p for (p,v) in [(pred,value) for (pred,value) in list(zip(list_predictions,argratio)) if value != -999]]
            }

print("Total length noisy test set Cross-Domain")
print("maxsim1",len(PEPs["aurc_cr_predictions_test_001.json_PEP.tsv"]["list_predictions_noisy_maxsim1"]))
print("maxsim2",len(PEPs["aurc_cr_predictions_test_001.json_PEP.tsv"]["list_predictions_noisy_maxsim2"]))
print("argratio",len(PEPs["aurc_cr_predictions_test_001.json_PEP.tsv"]["list_predictions_noisy_argratio"]))

token_maxsim1 = []
token_maxsim2 = []
token_argratio = []
tokencrf_maxsim1 = []
tokencrf_maxsim2 = []
tokencrf_argratio = []
sentence_maxsim1 = []
sentence_maxsim2 = []
sentence_argratio = []
for filename, d4 in PEPs.items():
    print("---")
    print(filename)
    print("maxsim1\t",stats.pointbiserialr(d4["list_predictions_noisy_maxsim1"],d4["maxsim1"]))
    print("maxsim2\t",stats.pointbiserialr(d4["list_predictions_noisy_maxsim2"],d4["maxsim2"]))
    print("argratio\t",stats.pointbiserialr(d4["list_predictions_noisy_argratio"],d4["argratio"]))
    correlation_maxsim1, _ = stats.pointbiserialr(d4["list_predictions_noisy_maxsim1"],d4["maxsim1"])
    correlation_maxsim2, _ = stats.pointbiserialr(d4["list_predictions_noisy_maxsim2"],d4["maxsim2"])
    correlation_argratio, _ = stats.pointbiserialr(d4["list_predictions_noisy_argratio"],d4["argratio"])
    if "_crf_" in filename:
        tokencrf_maxsim1.append(correlation_maxsim1)
        tokencrf_maxsim2.append(correlation_maxsim2)
        tokencrf_argratio.append(correlation_argratio)
    elif "_sentence_" in filename:
        sentence_maxsim1.append(correlation_maxsim1)
        sentence_maxsim2.append(correlation_maxsim2)
        sentence_argratio.append(correlation_argratio)
    else:
        token_maxsim1.append(correlation_maxsim1)
        token_maxsim2.append(correlation_maxsim2)
        token_argratio.append(correlation_argratio)
    print("---")

def meanstd(score_name, list_of_scores):
    return score_name+"|mean,std|\t"+str(numpy.mean(list_of_scores))+"\t"+str(numpy.std(list_of_scores))

m = {
        "token_maxsim1":token_maxsim1,
        "token_maxsim2":token_maxsim2,
        "token_argratio":token_argratio,
        "tokencrf_maxsim1":tokencrf_maxsim1,
        "tokencrf_maxsim2":tokencrf_maxsim2,
        "tokencrf_argratio":tokencrf_argratio,
        "sentence_maxsim1":sentence_maxsim1,
        "sentence_maxsim2":sentence_maxsim2,
        "sentence_argratio":sentence_argratio
        }

for score_name, list_of_scores in m.items():
    print(meanstd(score_name, list_of_scores))