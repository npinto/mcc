# -*- coding: utf-8 -*-
"""Maximum Correlation Classifier
with scikits-learn like API

Training of this classifier consists of creating C "classification vectors"
(where C is the number of categories/classes/conditions used in the analysis),
and each classification vector is simply the mean of all the training data from
that category (thus each classification vector is a point in Rm, where m is the
number of variables/features). To assess to which class a test point belongs,
the Pearson's correlation coefficient (i.e. dot product between two zero-mean
and unit-euclidean-norm vectors) is calculated between the test point and each
classification vector; a test data point is classified as belonging to the
category Ci, if the correlation coefficient between the test point and the
classification vector of class Ci is greater than the correlation coefficient
between the test point and the classification vector of any other class.

There are several reasons why you may want to use a correlation
coefficient-based classifier (adapted from Meyers et al. 2008):

1. Because this is a linear classifier, applying the classifier is analogous to
the integration of presynaptic activity through synaptic weights; thus decoding
accuracy can be thought of as indicative of the information available to the
postsynaptic targets of the neurons being analyzed.

2. Computation with this classifier is very fast, and it has empirically given
classification accuracies on neural data that are comparable to more
sophisticated classifiers such as regularized least squares, support vector
machines and Poisson naive Bayes classifiers, which we have tested on this and
other data sets (see Meyers et al. 2008, Supplemental Fig. S2).

3. This classifier is invariant to scalar addition and multiplication of the
data, which might be useful for comparing data across different time periods in
which the mean firing rate of the population might have changed. And finally,
this classifier has no free adjustable parameters (that are not determined by
the data) which simplifies the training procedure.
(text adapted from Meyers et al. 2008)

References:
-----------
Wikipedia: Correlation and dependence
http://en.wikipedia.org/wiki/Correlation_and_dependence

Dynamic Population Coding of Category Information in Inferior Temporal and
Prefrontal Cortex
Ethan M. Meyers, David J. Freedman, Gabriel Kreiman, Earl K. Miller, and Tomaso
Poggio
J Neurophysiol 100: 1407-1419, 2008.
doi:10.1152/jn.90248.2008.
http://monkeylogic.uchicago.edu/Meyers_et_al_2008.pdf

Backlog:
--------
* TODO: multi-class formulation
* TODO: learn a threshold for predict()
"""


import numpy as np
from numpy import linalg
from itertools import izip


# -----------------------------------------------------------------------------
# -- Main class
# -----------------------------------------------------------------------------
class MaximumCorrelationClassifier(object):

    def __init__(self, n_features, n_classes=2, dtype=np.float32):

        if n_classes != 2:
            raise NotImplementedError("Only two-class problems are supported")

        self.n_features = n_features
        self.n_classes = n_classes
        self.dtype = dtype

        self._unormalized_weights_pos = np.zeros((n_features), dtype=dtype)
        self._unormalized_weights_neg = np.zeros((n_features), dtype=dtype)

        self._n_pos = 0
        self._n_neg = 0

        self._partial_fitting = False

    def partial_fit(self, X, y):
        """Incrementally learn the decision boundary.

        If you call this method multiple times, weights will be changed
        incrementally.
        """

        unormalized_weights_pos = self._unormalized_weights_pos
        unormalized_weights_neg = self._unormalized_weights_neg

        n_pos = self._n_pos
        n_neg = self._n_neg

        for obs, label in izip(X, y):
            if label > 0:
                n_pos += 1
                step_size = 1. / n_pos
                unormalized_weights_pos *= (1 - step_size)
                unormalized_weights_pos += step_size * obs
            else:
                n_neg += 1
                step_size = 1. / n_neg
                unormalized_weights_neg *= (1 - step_size)
                unormalized_weights_neg += step_size * obs

        self._unormalized_weights_pos = unormalized_weights_pos
        self._unormalized_weights_neg = unormalized_weights_neg
        self._n_pos = n_pos
        self._n_neg = n_neg

        self._partial_fitting = True

    def fit(self, X, y):
        """One-shot learning of the decision boundary.
        """

        if self._partial_fitting:
            raise RuntimeError("It appears that partial_fit() "
                               "has been called before, "
                               "sorry but you can't use fit() anymore. "
                               "You may want to consider instantiating "
                               "a new '%s' object"
                               % self.__class__.__name__)

        pos_idx = y > 0

        pos = X[pos_idx]
        if pos.size > 0:
            self._unormalized_weights_pos = pos.mean(0)
            self._n_pos = pos.shape[0]

        neg = X[-pos_idx]
        if neg.size > 0:
            self._unormalized_weights_neg = neg.mean(0)
            self._n_neg = neg.shape[0]

    def decision_function(self, X):

        # normalize X
        X_normalized = X - X.mean(1)[:, None]
        lengths = np.sqrt((X_normalized ** 2.).sum(1))
        lengths[lengths == 0] = 1
        X_normalized /= lengths[:, None]

        # combine and normalize weights
        unormalized_weights = self._unormalized_weights_pos - \
                self._unormalized_weights_neg
        weights = zero_mean_unit_length(unormalized_weights)

        return np.dot(X_normalized, weights)

    def predict(self, X):
        return np.sign(self.decision_function(X))


# -----------------------------------------------------------------------------
# -- Helpers
# -----------------------------------------------------------------------------
def zero_mean_unit_length(vector):
    assert vector.ndim == 1
    out = vector - vector.mean()
    length = linalg.norm(out)
    if length > 0:
        out /= length
    return out
