import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack
from util.tsr_function import ContTable, information_gain
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfTransformer


class DistributionalRandomOversampling:
    """
    Distributional Random Oversampling (DRO) is an oversampling method to counter data imbalance in binary text
    classification. DRO generates new random minority-class synthetic documents by exploiting the distributional
    properties of the terms in the collection. The variability introduced by the oversampling method is enclosed in a
    latent space; the original space is replicated and left untouched.

    @inproceedings{moreo2016dro,
      title={Distributional random oversampling for imbalanced text classification},
      author={Moreo, Alejandro and Esuli, Andrea and Sebastiani, Fabrizio},
      booktitle={Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval},
      pages={805--808},
      year={2016},
      organization={ACM}
    }
    """

    def __init__(self, rebalance_ratio=0.2):
        """
        :param rebalance_ratio: the proportion of positive examples after resampling.
        """
        self.rebalance_ratio = rebalance_ratio
        self.latent_tfidf = TfidfTransformer()
        self.dummy=False

    def fit_transform(self, X, y, words_by_doc):
        """
        Fits the method and produces the oversampling of the matrix.
        :param X: a tfidf weighted co-occurrence matrix of shape (n_docs, n_feats)
        :param y: a binary ndarray of shape (n_docs)
        :param words_by_doc: a ndarray of shape (n_docs) indicating the number of latent terms to generate for each
        document, or an int if the same number is requested for all documents.
        :return: an oversampled matrix of shape (d, f), where d>n_docs is the new number of documents
        after resampling (i.e., after oversampling the minority class as to match the rebalance_ratio of positives), and
        f>n_feats is the enlarged space consisting of n_feats (the original space) + n_docs (the latent space, which
        coincides with the number of documents). If the prevalence of y>=rebalance_ratio, then the method does nothing
        and returns X and y since no undersampling of the majority class is performed (undersampling is believed to
        be detrimental for classification since it losses information).
        """
        if y.mean()>self.rebalance_ratio:
            self.dummy=True
            raise UserWarning(
                'the class has prevalence higher than the requested rebalance ratio (no transformation will be performed)'
            )
            return X, y


        # computes how many times each document has to be oversampled in order to match the requested rebalance ratio
        samples = self._samples_to_match_ratio(y)

        # obtains the weight matrix from the training set, that will be used to sample latent terms for synthetic docs
        self.weight_matrix = get_weight_matrix(X, y)

        O = self._oversampling_observed(X, samples)
        L = self._oversampling_latent(X, words_by_doc, samples)
        L = self.latent_tfidf.fit_transform(L)
        X_ = hstack([O,L])

        y = self._oversampling_observed(y, samples)

        return X_, y

    def transform(self, X, words_by_doc, samples):
        """
        Applies DRO to X.
        :param X: a tfidf weighted csr_matrix of shape (n_docs, n_feats)
        :param words_by_doc: a ndarray of shape (n_docs) indicating the number of latent terms to generate for each
        document, or an int if the same number is requested for all documents.
        :param samples: a ndarray of shape (n_docs) indicating the number of samples to generate from each document,
        or an int if the same number is requested for all documents.
        :return:
        """
        if self.dummy: return X
        assert hasattr(self, 'weight_matrix'), 'transform called before fit'
        samples = as_array_of_ints(samples, nD=X.shape[0])

        O = self._oversampling_observed(X, samples)
        L = self._oversampling_latent(X, words_by_doc, samples)
        L = self.latent_tfidf.transform(L)
        X_ = hstack([O, L])

        return X_

    def _oversampling_latent(self, X, words_by_doc, samples):
        words_by_doc = as_array_of_ints(words_by_doc, nD=X.shape[0])

        # computes the parameters of the multinomial distribution from which the latent terms will be drawn
        p = X.dot(self.weight_matrix).toarray()
        p = normalize(p, norm='l1', axis=1, copy=True)

        # sampling of latent terms
        latent_space = []
        for i,n in tqdm(list(enumerate(samples)), desc='oversampling'):
            for _ in range(n):
                latent_row = csr_matrix(np.random.multinomial(words_by_doc[i], pvals=p[i]))
                latent_space.append(latent_row)
        latent_space = vstack(latent_space)

        return  latent_space

    def _oversampling_observed(self, X, samples):
        # replicates elements of X as many times as indicated by the entries in samples
        nD = X.shape[0]
        observed_space = X[np.repeat(np.arange(nD), samples)]
        return observed_space

    def _samples_to_match_ratio(self, y):
        replicate = np.ones_like(y)
        nD = len(y)
        positives = y.sum()
        missing_positives = int((positives - nD*self.rebalance_ratio)/(self.rebalance_ratio - 1))
        multipla = int(missing_positives // positives)
        replicate[y==1] += multipla
        remaining = missing_positives - multipla*positives
        if remaining > 0:
            replicate[np.random.choice(np.argwhere(y==1).flatten(), remaining, replace=False)] += 1
        return replicate


def as_array_of_ints(val, nD):
    if isinstance(val, int):
        return np.full(nD, val)
    elif isinstance(val, list) or isinstance(val, np.ndarray):
        assert len(val) == nD, 'wrong shape (must be equal to the number of documents)'
        return np.asarray(val)
    raise ValueError('unexpected type. Should be int, list, or array of shape (n_docs)')


def feature_informativeness(X, y):
    X = csc_matrix(X)
    nD, nF = X.shape
    positives = y.sum()
    negatives = nD - positives

    # computes the 4-cell contingency tables for each feature
    TP = np.asarray((X[y == 1] > 0).sum(axis=0)).flatten()
    FN = positives - TP
    FP = np.asarray((X[y == 0] > 0).sum(axis=0)).flatten()
    TN = negatives - FP
    _4cell = [ContTable(tp=TP[i], tn=TN[i], fp=FP[i], fn=FN[i]) for i in range(nF)]

    # applies the tsr_function to the 4-cell counters
    return np.array(list(map(information_gain, _4cell)))


def get_weight_matrix(X, y):
    feat_info = feature_informativeness(X, y)
    Xnorm = normalize(X, norm='l1', axis=0, copy=True)
    return csr_matrix(Xnorm.multiply(feat_info)).T