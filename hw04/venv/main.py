from sklearn.model_selection import train_test_split
import numpy as np
import numpy
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
from catboost import CatBoostClassifier
import catboost


def gini(x):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return np.sum(proba * (1 - proba))


def entropy(x):
    _, counts = np.unique(x, return_counts=True)
    proba = counts / len(x)
    return -np.sum(proba * np.log2(proba))


def gain(left_y, right_y, criterion):
    y = np.concatenate((left_y, right_y))
    return criterion(y) - (criterion(left_y) * len(left_y) + criterion(right_y) * len(right_y)) / len(y)


# Task 1
class DecisionTreeLeaf:
    def __init__(self, data):
        names, counts = np.unique(data, return_counts=True)
        id = 0
        for i in range(len(counts)):
            if counts[i] > counts[id]:
                id = i
        self.y = names[id]
        self.proba = {}
        for i in range(len(names)):
            name = names[i]
            count = counts[i]
            self.proba[name] = count / len(data)


class DecisionTreeNode:
    def __init__(self, split_dim, left, right):
        self.split_dim = split_dim
        self.left = left
        self.right = right


class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        self.root = None
        self.criterion_str = criterion
        if criterion == "gini":
            self.criterion = gini
        else:
            self.criterion = entropy
        if max_depth is None:
            self.max_depth = len(X)
        else:
            self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        if max_features == "auto":
            self.max_features = round(len(X[0]) ** (1 / 2))
        else:
            self.max_features = int(max_features)
        self.X = X
        self.y = y
        self.bag = None

    def build(self, X, y, depth=0):
        if depth == self.max_depth:
            return DecisionTreeLeaf(y)
        if len(np.unique(X)) == 1:
            return DecisionTreeLeaf(y)

        best = None
        split_dim = None
        left_X = []
        left_y = []
        right_X = []
        right_y = []

        dims = []
        for i in range(len(X[0])):
            dims.append(i)
        numpy.random.shuffle(dims)
        dims = dims[:self.max_features]

        for dim in dims:
            left = []
            right = []
            for j in range(len(X)):
                if X[j][dim] == 0:
                    left.append(y[j])
                else:
                    right.append(y[j])

            if min(len(left), len(right)) < self.min_samples_leaf:
                continue

            result = gain(left, right, self.criterion)
            if best is None or result > best:
                best = result
                split_dim = dim

        if best is None:
            return DecisionTreeLeaf(y)

        for j in range(len(X)):
            if X[j][split_dim] == 0:
                left_X.append(X[j])
                left_y.append(y[j])
            else:
                right_X.append(X[j])
                right_y.append(y[j])

        return DecisionTreeNode(split_dim,
                                self.build(left_X, left_y, depth + 1),
                                self.build(right_X, right_y, depth + 1))

    def fit(self, X, y):
        self.root = self.build(X, y)
        self.bag = []
        n = len(X)
        self.bag_X = []
        self.bag_y = []
        for i in range(n):
            self.bag.append(numpy.random.randint(n))
            self.bag_X.append(X[self.bag[i]])
            self.bag_y.append(y[self.bag[i]])
        self.root = self.build(self.bag_X, self.bag_y)

    def predict_proba(self, X):
        result = []
        for x in X:
            result.append(self.get_proba(x, self.root))
        return result

    def get_proba(self, x, node):
        if isinstance(node, DecisionTreeLeaf):
            return node.proba
        if x[node.split_dim] == 0:
            return self.get_proba(x, node.left)
        else:
            return self.get_proba(x, node.right)

    def predict(self, X):
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]

    def err_oob(self):
        result = []
        oob_X = []
        oob_y = []
        for i in range(len(self.X)):
            if i not in self.bag:
                oob_X.append(self.X[i])
                oob_y.append(self.y[i])
        y_pred = self.predict(oob_X)
        err_count = 0
        ids = []
        for i in range(len(oob_y)):
            ids.append(i)
            if oob_y[i] != y_pred[i]:
                err_count += 1

        for i in range(len(oob_X[0])):
            numpy.random.shuffle(ids)
            new_X = []
            for j in range(len(oob_X)):
                new_X.append(oob_X[j])
                new_X[j][i] = oob_X[j][i]
            pred = self.predict(new_X)
            cur_err_count = 0
            for j in range(len(new_X)):
                if oob_y[j] != pred[j]:
                    cur_err_count += 1
            result.append(cur_err_count - err_count)
        return numpy.array(result) / len(oob_y)


# Task 2
class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.forest = None

    def fit(self, X, y):
        forest = []
        for i in range(self.n_estimators):
            forest.append(DecisionTree(X, y, self.criterion, self.max_depth,
                                       self.min_samples_leaf, self.max_features))
            forest[i].fit(X, y)
        self.forest = forest

    def predict(self, X):
        results = []
        for i in range(self.n_estimators):
            results.append(self.forest[i].predict(X))
        result = []
        for j in range(len(results[0])):
            count = {}
            for i in range(len(results)):
                if results[i][j] in count:
                    count[results[i][j]] += 1
                else:
                    count[results[i][j]] = 1
            res_j = -1
            for key, value in count.items():
                if res_j == -1 or count[res_j] < value:
                    res_j = key
            result.append(res_j)
        return result


# Task 3
def feature_importance(rfc):
    result = rfc.forest[0].err_oob()
    for i in range(1, rfc.n_estimators):
        result = sum(result, rfc.forest[i].err_oob())
    return result / rfc.n_estimators


def most_important_features(importance, names, k=20):
    # Выводит названия k самых важных признаков
    idicies = np.argsort(importance)[::-1][:k]
    return np.array(names)[idicies]


def synthetic_dataset(size):
    X = [(np.random.randint(0, 2), np.random.randint(0, 2), i % 6 == 3,
          i % 6 == 0, i % 3 == 2, np.random.randint(0, 2)) for i in range(size)]
    y = [i % 3 for i in range(size)]
    return np.array(X), np.array(y)


X, y = synthetic_dataset(1000)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, y)
print("Accuracy:", np.mean(rfc.predict(X) == y))
print("Importance:", feature_importance(rfc))

#Task 4
def read_dataset(path):
    dataframe = pandas.read_csv(path, header=0)
    dataset = dataframe.values.tolist()
    random.shuffle(dataset)
    y_age = [row[0] for row in dataset]
    y_sex = [row[1] for row in dataset]
    X = [row[2:] for row in dataset]

    return np.array(X), np.array(y_age), np.array(y_sex), list(dataframe.columns)[2:]


X, y_age, y_sex, features = read_dataset("vk.csv")
X_train, X_test, y_age_train, y_age_test, y_sex_train, y_sex_test = train_test_split(X, y_age, y_sex, train_size=0.9)

rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_age_train)
print("Accuracy:", np.mean(rfc.predict(X_test) == y_age_test))
print("Most important features:")
for i, name in enumerate(most_important_features(feature_importance(rfc), features, 20)):
    print(str(i+1) + ".", name)

rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_sex_train)
print("Accuracy:", np.mean(rfc.predict(X_test) == y_sex_test))
print("Most important features:")
for i, name in enumerate(most_important_features(feature_importance(rfc), features, 20)):
    print(str(i+1) + ".", name)

X, y = synthetic_dataset(1000)
cb = CatBoostClassifier(loss_function='MultiClass')
cb.fit(X, y)
print("Accuracy:", np.mean(cb.predict(X) == y))
print("Importance:", cb.get_feature_importance(catboost.Pool(X, y)))

# Task 5
X, y_age, y_sex, features = read_dataset("vk.csv")
X_train, X_test, y_age_train, y_age_test, y_sex_train, y_sex_test = train_test_split(X, y_age, y_sex, train_size=0.9)
X_train, X_eval, y_age_train, y_age_eval, y_sex_train, y_sex_eval = train_test_split(X_train, y_age_train, y_sex_train, train_size=0.8)

cb = CatBoostClassifier(loss_function='MultiClass')
cb.fit(X_train, y_age_train)
print("Accuracy:", np.mean(cb.predict(X_test) == y_age_test))
print("Most important features:")
for i, name in enumerate(most_important_features(cb.get_feature_importance((catboost.Pool(X_train, y_age_train))), features, 10)):
    print(str(i+1) + ".", name)

cb = CatBoostClassifier(loss_function='MultiClass')
cb.fit(X_train, y_sex_train)
print("Accuracy:", np.mean(cb.predict(X_test) == y_sex_test))
print("Most important features:")
for i, name in enumerate(most_important_features(cb.get_feature_importance((catboost.Pool(X_train, y_sex_train))), features, 10)):
    print(str(i+1) + ".", name)
