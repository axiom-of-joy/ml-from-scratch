class LinearRegressor:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        N, self.p = X.shape
        self.beta = np.random.rand([1, self.p + 1])
        X = self._pad_ones(X)

        for _ in range(epochs):
            self.beta -= (self.lr / N) * np.dot((self.predict(X, pad=False) - y).T, X)

    def predict(self, X, pad=True):
        if X.shape[1] != self.p + (0 if pad else 1):
            raise ValueError
        if pad:
            X = self._pad_ones(X)
        return np.dot(X, self.beta)

    def _pad_ones(self, X):
        n = X.shape[0]
        X = np.concatenate([np.ones([N, 1]), X])
        return X

