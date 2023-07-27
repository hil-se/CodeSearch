import numpy as np
import tensorflow as tf

class Metrics:

    def __init__(self, trues, preds):
        self.y = trues
        self.y_pred = preds
        self.conf = self.confusion()
        self.pos = 0
        self.neg = 0
        self.min_p = 2
        self.min_n = 2
        self.max_p = -2
        self.max_n = -2
        self.mrr_score = 0
        self.map = 0

    def confusion(self):
        conf = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
        for i in range(len(self.y)):
            if self.y[i]==0 and self.y_pred[i]==0:
                conf['tn']+=1
            elif self.y[i]==1 and self.y_pred[i]==1:
                conf['tp'] += 1
            elif self.y[i]==0 and self.y_pred[i]==1:
                conf['fp'] += 1
            elif self.y[i]==1 and self.y_pred[i]==0:
                conf['fn'] += 1
        return conf

    def accuracy(self):
        return (self.conf['tn']+self.conf['tp'])/len(self.y)

    def f1(self):
        try:
            f1 = self.tpr()*self.prec()*2/(self.tpr()+self.prec())
        except:
            f1 = 0.0
        return f1

    def f2(self):
        try:
            f2 = (5 * self.tpr() * self.prec()) / ((4 * self.prec()) + self.tpr())
        except:
            f2 = 0.0
        return f2

    def tpr(self):
        try:
            return self.conf['tp']/(self.conf['tp']+self.conf['fn'])
        except:
            return 0.0


    def prec(self):
        try:
            return self.conf['tp'] / (self.conf['tp'] + self.conf['fp'])
        except:
            return 0.0

    def calcCosineSim(self, preds):
        preds = preds.numpy()
        pos = 0
        neg = 0
        for idx, x in np.ndenumerate(preds):
            i = idx[0]
            j = idx[1]
            if i==j:
                pos+=x
                self.max_p = max(self.max_p, x)
                self.min_p = min(self.min_p, x)
            else:
                neg+=x
                self.max_n = max(self.max_n, x)
                self.min_n = min(self.min_n, x)
        m, n = preds.shape
        pos/=m
        neg/=((m*n)-m)
        self.pos = pos
        self.neg = neg


    def mrr(self, source_embs, target_embs, preds):
        sources = []
        targets = []
        for i in range(len(source_embs)):
            thisTensor = source_embs[i]
            if isInList(sources, thisTensor)==False:
                sources.append(thisTensor)
        for i in range(len(target_embs)):
            thisTensor = target_embs[i]
            if isInList(targets, thisTensor)==False:
                targets.append(thisTensor)
        links = []
        ln = len(source_embs)
        for i in range(ln):
            s = source_embs[i]
            t = target_embs[i]
            s_ind = findIndex(sources, s)
            t_ind = findIndex(targets, t)
            links.append((s_ind, t_ind))
        preds = preds.numpy()
        res = {}
        for idx, score in np.ndenumerate(preds):
            i = idx[0]
            j = idx[1]
            if i not in res:
                res[i] = []
            res[i].append((j, score))
        corr = 0
        for key, val in res.items():
            res[key] = sorted(val, key=lambda x: float(x[1]), reverse=True)
            rank = self.getRank(res[key], key, links)
            if rank != 0:
                corr += (1 / rank)
        self.mrr_score = (corr / len(res))

    def getRank(self, listOfPairs, index, links):
        for i in range(len(listOfPairs)):
            targetIndex = listOfPairs[i][0]
            if index==targetIndex or (index, targetIndex) in links:
                return (i+1)
        return 0

    def mrrForSingleLinks(self, preds):
        preds = preds.numpy()
        res = {}
        for idx, score in np.ndenumerate(preds):
            i = idx[0]
            j = idx[1]
            if i not in res:
                res[i] = []
            res[i].append((j, score))
        corr = 0
        for key, val in res.items():
            res[key] = sorted(val, key=lambda x: float(x[1]), reverse=True)
            rank = self.getRankForSingleLinks(res[key], key)
            if rank!=0:
                corr+=(1/rank)
        self.mrr_score = (corr/len(res))

    def getRankForSingleLinks(self, listOfPairs, index):
        for i in range(len(listOfPairs)):
            targetIndex = listOfPairs[i][0]
            if index==targetIndex: # Change this condition to check if pair is in links
                return (i+1)
        return 0

    def getCosMat(self, preds, rank):
        preds = preds.numpy()
        res = {}
        for idx, score in np.ndenumerate(preds):
            i = idx[0]
            j = idx[1]
            if i not in res:
                res[i] = []
            res[i].append((j, score))
        corr = 0
        mat = [[0 for n in range(rank)] for m in range(len(res))]
        for key, val in res.items():
            res[key] = sorted(val, key=lambda x: float(x[1]), reverse=True)
            val = res[key]
            for i in range(rank):
                if key == val[i][0]:
                    mat[key][i] = 1
                else:
                    mat[key][i] = 0
        return mat

    def cosineSimTrue(self):
        return self.pos

    def cosineSimFalse(self):
        return self.neg

    def getMaxP(self):
        return self.max_p

    def getMaxN(self):
        return self.max_n

    def getMinP(self):
        return self.min_p

    def getMinN(self):
        return self.min_n

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

class Metrics_Retrieval:
    def __init__(self, ranks):
        self.ranks = ranks

    def mavP(self):
        return np.mean([np.mean([np.mean(self.ranks[i, :j + 1]) for j in range(self.ranks.shape[1])]) for i in
                        range(self.ranks.shape[0])])

    def mAK(self):
        return np.mean([np.max(self.ranks[i]) for i in range(self.ranks.shape[0])])

    def mavA(self):
        return np.mean([np.mean([np.max(self.ranks[i, :j + 1]) for j in range(self.ranks.shape[1])]) for i in
                        range(self.ranks.shape[0])])

def MAP(ranks):
    return np.mean([np.mean([np.mean(row[:i+1]) for i in range(sum(row))]) for row in ranks])

def WebQueryMetrics(scores, labels, threshold):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    total_len = len(labels)
    for i in range(total_len):
        if scores[i]>=threshold:
            if labels[i]==1:
                TP+=1
            else:
                FP+=1
        else:
            if labels[i]==1:
                FN+=1
            else:
                TN+=1
    acc = (TP+TN)/total_len
    recall = 0
    precision = 0
    f1 = 0
    if (TP+FN)!=0:
        recall = TP/(TP+FN)
    if (TP+FP)!=0:
        precision = TP/(TP+FP)
    if (recall+precision)!=0:
        f1 = (2*recall*precision)/(recall+precision)
    return acc, recall, precision, f1

def getAvgCosineScores(scores, labels):
    total_len = len(scores)
    pos_score = 0
    neg_score = 0
    num_pos = 0
    for i in range(total_len):
        if labels[i]==1:
            pos_score+=scores[i]
            num_pos+=1
        else:
            neg_score+=scores[i]
    pos_score = pos_score/num_pos
    neg_score = neg_score/(total_len-num_pos)
    return pos_score, neg_score