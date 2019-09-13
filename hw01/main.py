import math

import matplotlib.pyplot as plt
import numpy
import pandas
import heapq


def read_cancer_dataset(path_to_csv):
    labels = pandas.read_csv(path_to_csv, usecols=["label"]).values.tolist()
    data = pandas.read_csv(path_to_csv,
                           usecols=lambda column: column not in ["label"],
                           ).values.tolist()
    return data, numpy.concatenate(labels)


def read_spam_dataset(path_to_csv):
    labels = pandas.read_csv(path_to_csv, usecols=["label"]).values.tolist()
    data = pandas.read_csv(path_to_csv,
                           usecols=lambda column: column not in ["label"],
                           ).values.tolist()
    return data, numpy.concatenate(labels)


def train_test_split(X, y, ratio):
    train = math.floor(len(X) * ratio)
    return X[:train], y[:train], X[train:], y[train:]


def get_set_of_labels(list):
    result = []
    for element in list:
        exists = False
        for existing in result:
            if element == existing:
                exists = True
        if (not exists):
            result.append(element)
    result.sort()
    return result


def get_precision(y_pred, y_true):
    classes = get_set_of_labels(y_true)
    result = []
    for current_class in classes:
        count = 0
        positive = 0
        for j in range(0, len(y_pred)):
            if y_pred[j] == current_class:
                count += 1
                if y_true[j] == y_pred[j]:
                    positive += 1
        if (count == 0):
            result.append(1)
        else:
            result.append((count - positive) / count)
    return result


def get_recall(y_pred, y_true):
    classes = get_set_of_labels(y_true)
    result = []

    for current_class in classes:
        count = 0
        positive = 0
        for j in range(0, len(y_true)):
            if y_true[j] == current_class:
                count += 1
                if y_true[j] == y_pred[j]:
                    positive += 1
        result.append((count - positive) / count)

    return result


def get_accuracy(y_pred, y_true):
    positive = 0
    for i in range(0, len(y_true)):
        if y_true[i] == y_pred[i]:
            positive += 1
    return positive / len(y_true)


def get_precision_recall_accuracy(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    accuracy = get_accuracy(y_pred, y_true)
    return precision, recall, accuracy


def plot_precision_recall(X_train, y_train, X_test, y_test, max_k=30):
    ks = list(range(1, max_k + 1))
    classes = len(numpy.unique(list(y_train) + list(y_test)))
    precisions = [[] for _ in range(classes)]
    recalls = [[] for _ in range(classes)]
    accuracies = []
    for k in ks:
        classifier = KNearest(k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        precision, recall, acc = get_precision_recall_accuracy(y_pred, y_test)
        for c in range(classes):
            precisions[c].append(precision[c])
            recalls[c].append(recall[c])
        accuracies.append(acc)

    def plot(x, ys, ylabel, legend=True):
        plt.figure(figsize=(12, 3))
        plt.xlabel("K")
        plt.ylabel(ylabel)
        plt.xlim(x[0], x[-1])
        plt.ylim(numpy.min(ys) - 0.01, numpy.max(ys) + 0.01)
        for cls, cls_y in enumerate(ys):
            plt.plot(x, cls_y, label="Class " + str(cls))
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.show()

    plot(ks, recalls, "Recall")
    plot(ks, precisions, "Precision")
    plot(ks, [accuracies], "Accuracy", legend=False)


def plot_roc_curve(X_train, y_train, X_test, y_test, max_k=30):
    positive_samples = sum(1 for y in y_test if y == 0)
    ks = list(range(1, max_k + 1))
    curves_tpr = []
    curves_fpr = []
    colors = []
    for k in ks:
        colors.append([k / ks[-1], 0, 1 - k / ks[-1]])
        knearest = KNearest(k)
        knearest.fit(X_train, y_train)
        p_pred = [p[0] for p in knearest.predict_proba(X_test)]
        tpr = []
        fpr = []
        for w in numpy.arange(-0.01, 1.02, 0.01):
            y_pred = [(0 if p > w else 1) for p in p_pred]
            tpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt == 0) / positive_samples)
            fpr.append(
                sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt != 0) / (len(y_test) - positive_samples))
        curves_tpr.append(tpr)
        curves_fpr.append(fpr)
    plt.figure(figsize=(7, 7))
    for tpr, fpr, c in zip(curves_tpr, curves_fpr, colors):
        plt.plot(fpr, tpr, color=c)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.show()


class KDTree:
    def __init__(self, X, leaf_size=40):
        self.tree = Tree(X, leaf_size, 0)

    # Возвращает i если return_distance = False, иначе возвращает пару (i, d)
    # i - массив ближайших соседей.
    #   Каждый элемент массива - список из k индексов элементов массива,
    #   на котором построено дерево в методе __init__
    # d - массив расстояний до каждого ближайшего соседа.
    #   Должен иметь такой же shape, как массив i
    def query(self, X, k=1, return_distance=True):
        result = []
        result_distance = []
        for i in range(0, len(X)):
            heap = []
            self.tree.query(X[i], heap, 0, k)
            point_result = []
            point_distance = []
            for _ in range(0, k):
                dist, point = heapq.heappop(heap)
                point_result.append(point)
                point_distance.append(dist)
            result.append(point_result)
            result_distance.append(point_distance)

        if return_distance:
            return result, result_distance

        return result


class Tree:
    def __init__(self, X, leaf_size, dimension, left_index=0):
        self.points = X
        self.left_index = left_index
        self.right_index = left_index + len(X) - 1
        self.leaf_size = leaf_size
        if len(X) <= leaf_size:
            return
        else:
            X = sorted(X, key=lambda x: x[dimension])
            self.points = X
            next_dimension = (dimension + 1) % len(X[0])
            half = len(X) // 2
            self.left = Tree(X[:len(X) - half - 1], leaf_size, next_dimension, 0)
            self.right = Tree(X[half:], leaf_size, next_dimension, half)

    def query(self, point, heap, dimension, k=1):
        if len(self.points) <= self.leaf_size:
            for i in range(0, len(self.points)):
                heapq.heappush(heap, {distance(point, self.points[i]), i + self.left_index})
            return
        half = len(self.points) // 2
        next_dimension = (dimension + 1) % len(point)
        split_line = (self.points[half - 1][dimension] + self.points[half][dimension]) / 2
        in_left = (point[dimension] <= split_line)
        if in_left:
            self.left.query(point, heap, next_dimension, k)
        else:
            self.right.query(point, heap, next_dimension, k)

        if len(heap) < k:
            if in_left:
                self.right.query(point, heap, next_dimension, k)
            else:
                self.left.query(point, heap, next_dimension, k)

        k_distance = distance(point, get_kth_point(min(k, len(heap)), heap))
        if k_distance > abs(point[dimension] - split_line):
            if in_left:
                self.right.query(point, heap, next_dimension, k)
            else:
                self.left.query(point, heap, next_dimension, k)


def get_kth_point(k, heap):
    top = []
    for i in range(0, k - 1):
        top.append(heapq.heappop(heap))
    result = heapq.heappop(heap)
    heapq.heappush(heap, result)
    for i in range(0, k - 1):
        heapq.heappush(heap, top[i])
    return result


def distance(x, y):
    return math.sqrt(sum([(x1 - y1) ** 2 for x1, y1 in zip(x, y)]))


def true_closest(X_train, X_test, k):
    result = []
    for x0 in X_test:
        bests = list(sorted([(i, numpy.linalg.norm(x - x0)) for i, x in enumerate(X_train)], key=lambda x: x[1]))
        bests = [i for i, d in bests]
        result.append(bests[:min(k, len(bests))])
    return result


class KNearest:
    def __init__(self, n_neighbors=5, leaf_size=30):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.tree = None
        self.classes = 0
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = get_set_of_labels(y)
        self.tree = KDTree(X, self.leaf_size)

    def predict_proba(self, X):
        # Возвращает матрицу, в которой строки соответствуют элементам X, а столбцы - классам.
        # На пересечении строки и столбца должна быть указана вероятность того, что элемент относится к классу
        #
        # Вероятность рассчитывается как
        # количество ближайших соседей с данным классом деленное на общее количество соседей
        result = []
        for i in range(0, len(X)):
            point_result = self.tree.query([X[i]], self.n_neighbors, False)
            point_result = point_result[0]
            point_probs = []
            for current_class in self.classes:
                positive = 0
                for j in range(0, self.n_neighbors):
                    if (self.y[int(point_result[j])] == current_class):
                        positive += 1
                point_probs.append(positive / self.n_neighbors)
            result.append(point_probs)
        return result

    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)


X_train = numpy.random.randn(100, 3)
X_test = numpy.random.randn(10, 3)
tree = KDTree(X_train, leaf_size=2)
predicted = tree.query(X_test, k=4, return_distance=False)
true = true_closest(X_train, X_test, k=4)

if numpy.sum(numpy.abs(numpy.array(numpy.array(predicted).shape) - numpy.array(numpy.array(true).shape))) != 0:
    print("Wrong shape")
else:
    errors = sum([1 for row1, row2 in zip(predicted, true) for i1, i2 in zip(row1, row2) if i1 != i2])
    if errors > 0:
        print("Encounted", errors, "errors")

X, y = read_spam_dataset("spam.csv")
X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
plot_precision_recall(X_train, y_train, X_test, y_test, max_k=20)
plot_roc_curve(X_train, y_train, X_test, y_test, max_k=20)
