import numpy as np
import copy
from sklearn.datasets import make_blobs, make_moons


class Module:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, d):
        raise NotImplementedError()

    def update(self, alpha):
        pass


class Linear(Module):
    def __init__(self, in_features, out_features):
        t = 1 / np.sqrt(in_features)
        self.w = np.random.uniform(-t, t, (in_features, out_features))
        self.b = np.zeros(out_features)
        self.x = None
        self.d = None

    def forward(self, x):
        self.x = np.dot(x, self.w) + self.b #x
        return self.x #np.dot(x, self.w) + self.b

    def backward(self, d):
        self.d = d
        return np.dot(d, self.w.T)

    def update(self, alpha):
        self.w -= alpha * np.dot(self.x.T, self.d)
        self.b -= alpha * self.d.mean(axis = 0) * self.x.shape[0]


class ReLU(Module):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = np.maximum(0, x)
        return self.x

    def backward(self, d):
        return d * (self.x > 0)


class Softmax(Module):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = np.exp(x) / sum(np.exp(x))
        return self.x

    def backward(self, d):
        return d * self.x * (1 - self.x)


class MLPClassifier:
    def __init__(self, modules, epochs=40, alpha=0.01):
        self.modules = modules + [Softmax()]
        self.epochs = epochs
        self.alpha = alpha

    def fit(self, X, y):
        n = len(self.modules)
        for i in range(self.epochs):
            front = X.copy()
            for i in range(n):
                front = self.modules[i].forward(front)

            size = len(X)
            ans = np.zeros((size, len(np.unique(y))))
            ans[np.arange(size), y] = 1
            back = (front - ans) / size

            for i in range(n):
                current = self.modules[n - i - 1]
                back = current.backward(back)

    def predict_proba(self, X):
        result = X.copy()
        for i in range(len(self.modules)):
            result = self.modules[i].forward(result)
        return result

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

X, y = make_moons(400, noise=0.075)
X_test, y_test = make_moons(400, noise=0.075)

best_acc = 0
for _ in range(25):
    p = MLPClassifier([
        Linear(2, 40),
        ReLU(),
        Linear(40, 120),
        Linear(120, 80),
        Linear(80, 80),
        ReLU(),
        Linear(80, 40),
        Linear(40, 2),
    ])

    p.fit(X, y)
    best_acc = max(np.mean(p.predict(X_test) == y_test), best_acc)
print("Accuracy", best_acc)


X, y = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
X_test, y_test = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5], [-2.5, 3]])
best_acc = 0
for _ in range(25):
    p = MLPClassifier([
        Linear(2, 80),
        ReLU(),
        Linear(80, 120),
        ReLU(),
        Linear(120, 80),
        ReLU(),
        Linear(80, 3)
    ])

    p.fit(X, y)
    best_acc = max(np.mean(p.predict(X_test) == y_test), best_acc)
print("Accuracy", best_acc)