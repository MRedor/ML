from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
import matplotlib.pyplot as plt
import matplotlib


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
    def __init__(self, split_dim, split_value, left, right):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right


class DecisionTreeClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1):
        self.root = None
        self.criterion_str = criterion
        if criterion == "gini":
            self.criterion = gini
        else:
            self.criterion = entropy
        if max_depth is None:
            self.max_depth = 0
        else:
            self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def build(self, X, y, depth=0):
        if depth == self.max_depth:
            return DecisionTreeLeaf(y)
        if len(np.unique(X)) == 1:
            return DecisionTreeLeaf(y)

        best = None
        split_dim = None
        split_value = None
        left_X = []
        left_y = []
        right_X = []
        right_y = []

        data = list(enumerate(X))

        for dim in range(len(X[0])):
            data.sort(key=lambda x: x[1][dim])
            for i in range(len(X)):
                split = X[i][dim]
                left = []
                right = []
                for j in range(len(X)):
                    if data[j][1][dim] < split:
                        left.append(y[data[j][0]])
                    else:
                        right.append(y[data[j][0]])

                if min(len(left), len(right)) < self.min_samples_leaf:
                    continue

                result = gain(left, right, self.criterion)
                if best is None or result > best:
                    best = result
                    split_dim = dim
                    split_value = split
        if best is None:
            return DecisionTreeLeaf(y)

        data.sort(key=lambda x: x[1][split_dim])
        for j in range(len(X)):
            if data[j][1][split_dim] < split_value:
                left_X.append(X[data[j][0]])
                left_y.append(y[data[j][0]])
            else:
                right_X.append(X[data[j][0]])
                right_y.append(y[data[j][0]])

        left = DecisionTreeClassifier(self.criterion_str, self.max_depth - 1, self.min_samples_leaf)
        left.build(left_X, left_y)
        right = DecisionTreeClassifier(self.criterion_str, self.max_depth - 1, self.min_samples_leaf)
        right.build(right_X, right_y)

        return DecisionTreeNode(split_dim, split_value,
                                self.build(left_X, left_y, depth + 1),
                                self.build(right_X, right_y, depth + 1))


    def fit(self, X, y):
        self.root = self.build(X, y)

    def predict_proba(self, X):
        result = []
        for x in X:
            result.append(self.get_proba(x, self.root))
        return result

    def get_proba(self, x, node):
        if isinstance(node, DecisionTreeLeaf):
            return node.proba
        if x[node.split_dim] < node.split_value:
            return self.get_proba(x, node.left)
        else:
            return self.get_proba(x, node.right)

    def predict(self, X):
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]

    def predict_explain(self, x):
        return self.explain(x, self.root)

    def explain(self, x, node):
        if isinstance(node, DecisionTreeLeaf):
            return node.y, ""
        if x[node.split_dim] < node.split_value:
            result, description = self.explain(x, node.left)
            description = "dimension " + str(node.split_dim) + ": " \
                          + str(x[node.split_dim]) + " < " + str(node.split_value) \
                          + "\n" + description
            return result, description
        else:
            result, description = self.explain(x, node.right)
            description = "dimension " + str(node.split_dim) + ": " \
                          + str(x[node.split_dim]) + " > " + str(node.split_value) \
                          + "\n" + description
            return result, description



def tree_depth(tree_root):
    if isinstance(tree_root, DecisionTreeNode):
        return max(tree_depth(tree_root.left), tree_depth(tree_root.right)) + 1
    else:
        return 1

def draw_tree_rec(tree_root, x_left, x_right, y):
    x_center = (x_right - x_left) / 2 + x_left
    if isinstance(tree_root, DecisionTreeNode):
        x_center = (x_right - x_left) / 2 + x_left
        x = draw_tree_rec(tree_root.left, x_left, x_center, y - 1)
        plt.plot((x_center, x), (y - 0.1, y - 0.9), c=(0, 0, 0))
        x = draw_tree_rec(tree_root.right, x_center, x_right, y - 1)
        plt.plot((x_center, x), (y - 0.1, y - 0.9), c=(0, 0, 0))
        plt.text(x_center, y, "x[%i] < %f" % (tree_root.split_dim, tree_root.split_value),
                horizontalalignment='center')
    else:
        plt.text(x_center, y, str(tree_root.y),
                horizontalalignment='center')
    return x_center

def draw_tree(tree, save_path=None):
    td = tree_depth(tree.root)
    plt.figure(figsize=(0.33 * 2 ** td, 2 * td))
    plt.xlim(-1, 1)
    plt.ylim(0.95, td + 0.05)
    plt.axis('off')
    draw_tree_rec(tree.root, -1, 1, td)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_test, p_pred):
    positive_samples = sum(1 for y in y_test if y == 0)
    tpr = []
    fpr = []
    for w in np.arange(-0.01, 1.02, 0.01):
        y_pred = [(0 if p.get(0, 0) > w else 1) for p in p_pred]
        tpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt == 0) / positive_samples)
        fpr.append(sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt != 0) / (len(y_test) - positive_samples))
    plt.figure(figsize = (7, 7))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.show()

def rectangle_bounds(bounds):
    return ((bounds[0][0], bounds[0][0], bounds[0][1], bounds[0][1]),
            (bounds[1][0], bounds[1][1], bounds[1][1], bounds[1][0]))

def plot_2d_tree(tree_root, bounds, colors):
    if isinstance(tree_root, DecisionTreeNode):
        if tree_root.split_dim:
            plot_2d_tree(tree_root.left, [bounds[0], [bounds[1][0], tree_root.split_value]], colors)
            plot_2d_tree(tree_root.right, [bounds[0], [tree_root.split_value, bounds[1][1]]], colors)
            plt.plot(bounds[0], (tree_root.split_value, tree_root.split_value), c=(0, 0, 0))
        else:
            plot_2d_tree(tree_root.left, [[bounds[0][0], tree_root.split_value], bounds[1]], colors)
            plot_2d_tree(tree_root.right, [[tree_root.split_value, bounds[0][1]], bounds[1]], colors)
            plt.plot((tree_root.split_value, tree_root.split_value), bounds[1], c=(0, 0, 0))
    else:
        x, y = rectangle_bounds(bounds)
        plt.fill(x, y, c=colors[tree_root.y] + [0.2])

def plot_2d(tree, X, y):
    plt.figure(figsize=(9, 9))
    colors = dict((c, list(np.random.random(3))) for c in np.unique(y))
    bounds = list(zip(np.min(X, axis=0), np.max(X, axis=0)))
    plt.xlim(*bounds[0])
    plt.ylim(*bounds[1])
    plot_2d_tree(tree.root, list(zip(np.min(X, axis=0), np.max(X, axis=0))), colors)
    for c in np.unique(y):
        plt.scatter(X[y==c, 0], X[y==c, 1], c=[colors[c]], label=c)
    plt.legend()
    plt.tight_layout()
    plt.show()


noise = 0.35
X, y = make_moons(1500, noise=noise)
X_test, y_test = make_moons(200, noise=noise)
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
tree.fit(X, y)
plot_2d(tree, X, y)
plot_roc_curve(y_test, tree.predict_proba(X_test))
draw_tree(tree)

X, y = make_blobs(1500, 2, centers=[[0, 0], [-2.5, 0], [3, 2], [1.5, -2.0]])
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
tree.fit(X, y)
plot_2d(tree, X, y)
draw_tree(tree)


def read_dataset(path):
    dataframe = pandas.read_csv(path, header=1)
    dataset = dataframe.values.tolist()
    random.shuffle(dataset)
    y = [row[0] for row in dataset]
    X = [row[1:] for row in dataset]
    return np.array(X), np.array(y)


X, y = read_dataset("train.csv")
dtc = DecisionTreeClassifier("entropy", 6, 10)
dtc.fit(X, y)


def predict_explain(dtc, X):
    result = []
    for x in X:
        result.append(dtc.predict_explain(x))
    return result


X, y = read_dataset("train.csv")
for pred_y, expl in predict_explain(dtc, X[:20]):
    print("Class:", pred_y)
    print("Explanation:", expl)
    print()