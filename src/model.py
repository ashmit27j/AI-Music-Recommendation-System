import numpy as np

class DecisionStumpMultiClass:
    """
    A decision stump for multi-class classification.
    Splits on one feature and threshold, assigns majority class left/right.
    """
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, y, sample_weight):
        m, n = X.shape
        classes = np.unique(y)
        min_error = float('inf')

        for feature_i in range(n):
            thresholds = np.unique(X[:, feature_i])
            for threshold in thresholds:
                left_mask = X[:, feature_i] < threshold
                right_mask = ~left_mask

                # Skip invalid splits
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                # Weighted votes for each class
                left_votes = {c: np.sum(sample_weight[left_mask] * (y[left_mask] == c)) for c in classes}
                right_votes = {c: np.sum(sample_weight[right_mask] * (y[right_mask] == c)) for c in classes}

                left_class = max(left_votes, key=left_votes.get)
                right_class = max(right_votes, key=right_votes.get)

                preds = np.where(X[:, feature_i] < threshold, left_class, right_class)
                error = np.sum(sample_weight * (preds != y))

                if error < min_error:
                    min_error = error
                    self.feature_index = feature_i
                    self.threshold = threshold
                    self.left_class = left_class
                    self.right_class = right_class

    def predict(self, X):
        return np.where(
            X[:, self.feature_index] < self.threshold,
            self.left_class,
            self.right_class
        )


class GradientBoostingMultiClass:
    """
    Gradient Boosting for multi-class classification using decision stumps.
    Implements one-vs-rest boosting logic with accurate step calculation and early stopping.
    """
    def __init__(self, n_estimators=20, learning_rate=0.1, early_stopping_rounds=None, verbose=True):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = {}  # class -> list of tuples (stump, gamma)
        self.classes = None
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None):
        self.classes = np.unique(y)
        m = X.shape[0]
        F = {c: np.zeros(m) for c in self.classes}  # One score per class per sample

        best_val_acc = 0
        rounds_since_improve = 0

        for c in self.classes:
            self.models[c] = []
            y_bin = np.where(y == c, 1, 0)

            for i in range(self.n_estimators):
                prob = self.sigmoid(F[c])
                residual = np.clip(y_bin - prob, -1, 1)
                sample_weight = np.abs(residual)

                stump = DecisionStumpMultiClass()
                stump.fit(X, y_bin, sample_weight)  # ✅ Fixed: use y_bin not y
                preds_bin = (stump.predict(X) == 1).astype(int)

                # Line search step size
                gamma = np.sum(residual * preds_bin) / (np.sum(preds_bin) + 1e-12)
                F[c] += self.learning_rate * gamma * preds_bin
                self.models[c].append((stump, gamma))

                if self.verbose and i % 5 == 0:
                    mean_res = np.mean(np.abs(residual))
                    print(f"Class {c}, Estimator {i}, mean residual: {mean_res:.4f}")

            # Early stopping (per-class)
            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                acc = np.mean(y_pred == y_val)
                if self.verbose:
                    print(f"[EarlyStopping] After class {c}: Validation Accuracy: {acc:.4f}")

                if acc > best_val_acc:
                    best_val_acc = acc
                    rounds_since_improve = 0
                else:
                    rounds_since_improve += 1

                if self.early_stopping_rounds and rounds_since_improve >= self.early_stopping_rounds:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break

    def predict(self, X):
        m = X.shape[0]
        logits = np.zeros((m, len(self.classes)))
        for idx, c in enumerate(self.classes):
            F_c = np.zeros(m)
            for stump, gamma in self.models[c]:
                F_c += self.learning_rate * gamma * (stump.predict(X) == 1)
            logits[:, idx] = F_c
        preds = self.classes[np.argmax(logits, axis=1)]
        return preds

    def predict_proba(self, X):
        m = X.shape[0]
        logits = np.zeros((m, len(self.classes)))
        for idx, c in enumerate(self.classes):
            F_c = np.zeros(m)
            for stump, gamma in self.models[c]:
                F_c += self.learning_rate * gamma * (stump.predict(X) == 1)
            logits[:, idx] = F_c
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
