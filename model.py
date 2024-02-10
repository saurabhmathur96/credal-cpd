import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from optimization import idm, monotonicity_violation, compute_bounds

class CredalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, target, parents, cardinality, s0=1, sn=10, prior_constraint=False):
        self.cardinality = cardinality
        self.s0 = s0 
        self.sn = sn 
        self.target = target 
        self.parents = parents
        self.prior_constraint = prior_constraint

    def fit(self, X, y, monotonicities, epsilon=0.001):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        parent_cards = [self.cardinality[name] for name in self.parents]
        target_card = self.cardinality[self.target]
        self.cases = np.array(list(np.ndindex(*parent_cards)))

        df = pd.DataFrame(X, columns = self.parents)
        df[self.target] = y
        value_counts = df[[self.target, *self.parents]].value_counts()
        
        counts = np.zeros((np.prod(parent_cards), target_card))
        for i, value in enumerate(self.cases):
            for j in range(target_card):
                counts[i, j] = value_counts.get((j, *value), 0) 

        for s in range(self.s0, self.sn+1):
            t_lower, t_upper, penalty_lower, penalty_upper = compute_bounds(self.target, self.parents, self.cardinality, 
                            self.cases, counts, monotonicities, 
                            s=s, epsilon=epsilon, tolerance=1e-8, prior_constraint=self.prior_constraint)
            if penalty_lower + penalty_upper < 2e-8:
                break
        self.counts = counts
        self.p_lower = idm(counts, s, t_lower)
        self.p_upper =  idm(counts, s, t_upper)
        # Return the classifier

        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        parent_cards = np.array([self.cardinality[name] for name in self.parents])
        factor = np.prod(parent_cards)/np.cumprod(parent_cards)
        indices = np.dot(X, factor).astype(int)
        y_pred = []
        # make prediction based on overlap
        for l, u in zip(self.p_lower[indices, :], self.p_upper[indices, :]):
            if l[1] > u[0]:
                y_pred.append(1)
            elif l[0] > u[1]:
                y_pred.append(0)
            else:
                y_pred.append(-1)
        return np.array(y_pred)