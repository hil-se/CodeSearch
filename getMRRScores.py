import numpy as np
import pandas as pd
from metrics import MAP
from DataReader import Reader
import tensorflow as tf
import time
pd.set_option('display.max_columns', None)


def load_data(lang=""):
    r = Reader(testset=True)
    r.load(lang.upper())
    data = pd.concat([r.source_series, r.target_series], axis=1)
    return data

def load_text_data(lang):
    data = pd.read_csv("Data/Texts/"+lang+"_test.csv")
    return data

def saveLinks(lang="CodeSearch_PYTHON"):
    text_data = load_text_data(lang)
    sources_list = text_data["Source"].unique()
    targets_list = text_data["Target"].unique()
    sources = {}
    targets = {}
    for i in range(len(sources_list)):
        sources[sources_list[i]]=i
    for i in range(len(targets_list)):
        targets[targets_list[i]]=i
    links = []
    ln = len(text_data.index)
    f = open("Results/Tensors/Links_"+lang.upper()+".txt", "w+")
    print("Generating links...")
    for index, row in text_data.iterrows():
        s = row["Source"]
        t = row["Target"]
        s_ind = sources[s]
        t_ind = targets[t]
        f.write(str((s_ind, t_ind)))
        links.append((s_ind, t_ind))
        f.write("\n")
    f.close()
    print("Saved links")
    return links

def loadLinks(lang="CodeSearch_PYTHON"):
    f = open("Results/Tensors/Links_"+lang.upper()+".txt", "r")
    lines = f.readlines()
    links = []
    for line in lines:
        row = line.replace("(", "")
        row = row.replace(")", "")
        row = row.replace("[", "")
        row = row.replace("]", "")
        numbers = row.split(", ")
        num1 = int(numbers[0])
        num2 = int(numbers[1])
        links.append((num1, num2))
    f.close()
    return links

def getRank(listOfPairs, index, links):
    for i in range(len(listOfPairs)):
        targetIndex = listOfPairs[i][0]
        # print(index, targetIndex)
        if index==targetIndex or (index, targetIndex) in links:
            return (i+1)
    return 0

def MRR(lang, links=None, duplicates=True):
    print("Loading data...")
    data = load_data(lang=lang)
    print("Data loaded.")
    sources = data["Source"].tolist()
    targets = data["Target"].tolist()
    links = loadLinks(lang=lang)
    currentTensor = None
    currentSims = None
    RR = 0
    acc = 0
    total_length = len(sources)
    start_time = time.time()
    abs_start_time = time.time()
    print("\nCalculating MRR score...\n")
    for i in range(total_length):
        if (i%500==0 and i!=0) or i==(total_length-1):
            end_time = time.time()
            print("Progress:", i, "/", total_length, ", Current MRR:", (RR/(i+1)), ", Accuracy:", acc/(i+1), ", Time since last update:", (round(end_time-start_time, 3)), "s")
            start_time = time.time()
        currentTensor = sources[i]
        currentSims = []
        for j in range(total_length):
            currentSims.append((j, np.matmul(currentTensor, targets[j])))
        currentSims = sorted(currentSims, key=lambda x: float(x[1]), reverse=True)
        targetIndex = currentSims[0][0]
        if duplicates==True:
            if targetIndex==i or (i, targetIndex) in links:
                acc+=1
        else:
            if targetIndex==i:
                acc+=1
        rank = getRank(currentSims, i, links)
        if rank!=0:
            RR += (1 / (rank))
    mrr_score = RR/total_length
    acc = acc/total_length
    end_time = time.time()
    fin_time = round(end_time-abs_start_time, 3)
    print("\nFinal MRR score:", mrr_score)
    print("Final accuracy:", acc)
    print("Total time taken:", fin_time, "s")
    return mrr_score, acc, fin_time


def getMRRScores(lang):
    results = []
    links = saveLinks(lang)
    mrr_score, acc, fin_time = MRR(lang=lang, links=links)
    return mrr_score, acc, fin_time
