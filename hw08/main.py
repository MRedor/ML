from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import spacy
from nltk.stem.snowball import SnowballStemmer

import re


def read_dataset(filename):
    file = open(filename, encoding="utf-8")
    x = []
    y = []
    for line in file:
        cl, sms = re.split("^b[\'\"](ham|spam)[\'\"],b[\'\"](.*)[\'\"]$", line)[1:3]
        x.append(sms)
        y.append(cl)
    return np.array(x, dtype=np.str), np.array(y, dtype=np.str)


def get_precision_recall_accuracy(y_pred, y_true):
    classes = np.unique(list(y_pred) + list(y_true))
    true_positive = dict((c, 0) for c in classes)
    true_negative = dict((c, 0) for c in classes)
    false_positive = dict((c, 0) for c in classes)
    false_negative = dict((c, 0) for c in classes)
    for c_pred, c_true in zip(y_pred, y_true):
        for c in classes:
            if c_true == c:
                if c_pred == c_true:
                    true_positive[c] = true_positive.get(c, 0) + 1
                else:
                    false_negative[c] = false_negative.get(c, 0) + 1
            else:
                if c_pred == c:
                    false_positive[c] = false_positive.get(c, 0) + 1
                else:
                    true_negative[c] = true_negative.get(c, 0) + 1
    precision = dict((c, true_positive[c] / (true_positive[c] + false_positive[c] + 1)) for c in classes)
    recall = dict((c, true_positive[c] / (true_positive[c] + false_negative[c])) for c in classes)
    accuracy = sum([true_positive[c] for c in classes]) / len(y_pred)
    return precision, recall, accuracy


def plot_precision_recall(X_train, y_train, X_test, y_test, bow_method, voc_sizes=range(4, 200, 5)):
    classes = np.unique(list(y_train) + list(y_test))
    precisions = dict([(c, []) for c in classes])
    recalls = dict([(c, []) for c in classes])
    accuracies = []
    for v in voc_sizes:
        bow = bow_method(X_train, voc_limit=v)
        X_train_transformed = bow.transform(X_train)
        X_test_transformed = bow.transform(X_test)
        classifier = NaiveBayes(0.001)
        classifier.fit(X_train_transformed, y_train)
        y_pred = classifier.predict(X_test_transformed)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for c in classes:
            precisions[c].append(precision[c])
            recalls[c].append(recall[c])
        accuracies.append(acc)

    def plot(x, ys, ylabel, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("Vocabulary size")
        plt.ylabel(ylabel)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(list(ys.values())) - 0.01, np.max(list(ys.values())) + 0.01)
        for c in ys.keys():
            plt.plot(x, ys[c], label="Class " + str(c))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    plot(voc_sizes, recalls, "Recall")
    plot(voc_sizes, precisions, "Precision")
    plot(voc_sizes, {"": accuracies}, "Accuracy", legend=False)


X, y = read_dataset("spam.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

#line = "b'ham',b\"Aight no rush, I'll ask jay\""
#line = "b'ham',b'Good Morning plz call me sir'"
#print(re.split("^b[\'\"](ham|spam)[\'\"],b[\'\"](.*)[\'\"]$", line))



#Task1

class NaiveBayes:
    def __init__(self, alpha):
        self.alpha = alpha  # Параметр аддитивной регуляризации
        self.classes = None
        self.counts = None
        self.words = []
        self.X = None
        self.probs = None

    def fit(self, X, y):
        self.classes, self.classCounts = np.unique(y, return_counts=True)
        self.X = X

        for x in X:
            self.words = np.concatenate([self.words, np.unique(x)])
            #self.words.append(np.unique(x))
        self.words = np.unique(self.words)
        self.wordCounts = []
        for word in self.words:
            result = []
            for x in X:
                cnt = 0
                for w in x:
                    if (word == w):
                        cnt += 1
                result.append(cnt)
            self.wordCounts.append(result)

        self.probs = []
        for c in self.classes:
            result = []
            denom = len(self.words)
            for i in range(len(self.words)):
                for j in range(len(X)):
                    if (y[j] != c):
                        continue
                    denom += self.wordCounts[i][j]
            for i in range(len(self.words)):
                sum = self.alpha
                for j in range(len(X)):
                    if (y[j] != c):
                        continue
                    sum += self.wordCounts[i][j]
                result.append(sum/denom)
            self.probs.append(result)

    def predict(self, X):
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]

    def log_proba(self, X):
        result = []
        for x in X:
            cur = []
            for i in range(len(self.classes)):
                c = self.classes[i]
                res = np.log(self.classCounts[i])
                for j in range(len(self.words)):
                    res += calc(x, self.words[j]) * np.log(self.probs[i][j])
                cur.append(res)
            result.append(cur)
        return result

def calc(x, w):
    cnt = 0
    for word in x:
        if word == w:
            cnt += 1
    return cnt


#Task2

class BoW:
    def __init__(self, X, voc_limit=1000):
        d = dict()
        for x in X:
            words = re.split("[^a-zA-Z]+", x.lower())
            for word in words:
                if word == "":
                    continue
                if word in d:
                    d[word] += 1
                else:
                    d[word] = 1
        self.words = sorted(d.keys(), key=lambda w: d[w], reverse=True)[:voc_limit]

    def transform(self, X):
        result = []
        for x in X:
            cur = np.zeros(len(self.words))
            words = re.split("[^a-zA-Z]+", x.lower())
            words, counts = np.unique(words, return_counts=True)
            for i in range(len(words)):
                if words[i] in self.words:
                    id = self.words.index(words[i])
                    cur[id] = counts[i]
            result.append(cur)
        return result


bow = BoW(X_train, voc_limit=500)
X_train_bow = bow.transform(X_train)
X_test_bow = bow.transform(X_test)

predictor = NaiveBayes(0.001)
predictor.fit(X_train_bow, y_train)
get_precision_recall_accuracy(predictor.predict(X_test_bow), y_test)

plot_precision_recall(X_train, y_train, X_test, y_test, BoW)

#Task3

class BowStem:
    def __init__(self, X, voc_limit=1000):
        self.stemmer = SnowballStemmer("english")
        d = dict()
        for x in X:
            words = re.split("[^a-zA-Z]+", self.stemmer.stem(x.lower()))
            for word in words:
                if word == "":
                    continue
                if word in d:
                    d[word] += 1
                else:
                    d[word] = 1
        self.words = sorted(d.keys(), key=lambda w: d[w], reverse=True)[:voc_limit]

    def transform(self, X):
        result = []
        for x in X:
            cur = np.zeros(len(self.words))
            words = re.split("[^a-zA-Z]+", self.stemmer.stem(x.lower()))
            words, counts = np.unique(words, return_counts=True)
            for i in range(len(words)):
                if words[i] in self.words:
                    id = self.words.index(words[i])
                    cur[id] = counts[i]
            result.append(cur)
        return result


bows = BowStem(X_train, voc_limit=500)
X_train_bows = bows.transform(X_train)
X_test_bows = bows.transform(X_test)

predictor = NaiveBayes(0.001)
predictor.fit(X_train_bows, y_train)
get_precision_recall_accuracy(predictor.predict(X_test_bows), y_test)

plot_precision_recall(X_train, y_train, X_test, y_test, BowStem)