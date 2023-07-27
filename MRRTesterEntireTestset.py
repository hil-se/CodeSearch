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

def equalTensors(a, b):
  c = tf.math.reduce_all(tf.equal(a,b))
  return c

def isInList(l, i):
  for j in range(len(l)):
    a = l[j]
    b = equalTensors(a, i)
    if b==True:
      return True
  return False

def findIndex(l, t):
  for i in range(len(l)):
    b = equalTensors(l[i], t)
    if b==True:
      return i
  return -1

def saveLinks(lang):
    data = load_data(lang=lang)
    source_embs = data["Source"].tolist()
    target_embs = data["Target"].tolist()
    sources = []
    targets = []
    print("Getting sources...")
    for i in range(len(source_embs)):
        thisTensor = source_embs[i]
        if isInList(sources, thisTensor) == False:
            sources.append(thisTensor)
    print("Getting targets...")
    for i in range(len(target_embs)):
        thisTensor = target_embs[i]
        if isInList(targets, thisTensor) == False:
            targets.append(thisTensor)
    links = []
    ln = len(source_embs)
    f = open("Results/Tensors/Links_"+lang.upper()+".txt", "w+")
    print("Generating links...")
    for i in range(ln):
        if (i % 200 == 0 and i != 0) or i == (ln - 1):
            print("Progress:", i+1, "/", ln)
        s = source_embs[i]
        t = target_embs[i]
        s_ind = findIndex(sources, s)
        t_ind = findIndex(targets, t)
        f.write(str((s_ind, t_ind)))
        links.append((s_ind, t_ind))
        f.write("\n")
    f.close()
    print("Saved links")
    return links

def loadLinks(lang):
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
    if duplicates==True:
        if links==None:
            links = loadLinks(lang=lang)
        print("Links loaded.")
    currentTensor = None
    currentSims = None
    RR = 0
    acc = 0
    total_length = len(sources)
    # total_length = 50
    start_time = time.time()
    abs_start_time = time.time()
    print("\nCalculating MRR score...\n")
    for i in range(total_length):
        if (i%200==0 and i!=0) or i==(total_length-1):
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
        # print("After sorting")
        # print(currentSims)
        rank = getRank(currentSims, i, links)
        # print("Rank:", rank)
        # print()
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

# languages = ["AdvTest_NoC"]
# results = []
# for lang in languages:
#     links = saveLinks(lang)
#     mrr_score, acc, fin_time = MRR(lang=lang, links=links)
#     results.append({"Language": lang, "MRR": mrr_score, "Accuracy": acc, "Testing time": fin_time})
# df = pd.DataFrame(results)
# print(df)
# df.to_csv("Results/AdvTest_NoC_results.csv", index=False)
