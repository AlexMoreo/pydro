from dro import DistributionalRandomOversampling
from data.dataset import Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, make_scorer


def classify_and_test(X, y, Xte, yte, average=1):
    svm = LinearSVC()
    svm.fit(X, y)
    # print(svm.best_params_)
    yte_ = svm.predict(Xte)
    if average>1:
        yte_ = yte_.reshape(-1,average).mean(axis=1)>0.5
    f1 = f1_score(yte, yte_)
    print(f'f1={f1:.3f}')


dataset = Dataset.load('reuters21578', pickle_path='./reuters.pickle')

Xtr, Xte = dataset.devel_raw, dataset.test_raw
ytr, yte = dataset.devel_target, dataset.test_target

# generate the co-occurrence matrices
counter = CountVectorizer(min_df=5)
Xtr = counter.fit_transform(Xtr)
Xte = counter.transform(Xte)
train_nwords = Xtr.sum(axis=1).getA().flatten()
test_nwords = Xte.sum(axis=1).getA().flatten()

# generate the tfidf matrices
tfidf = TfidfTransformer()
Xtr = tfidf.fit_transform(Xtr)
Xte = tfidf.transform(Xte)

# choose one category for binary classification
pos_cat = 6
ytr = ytr[:,pos_cat].toarray().flatten()
yte = yte[:,pos_cat].toarray().flatten()

# test the baseline (no oversampling)
positives = ytr.sum()
nD = len(ytr)
print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')

print('LinearSVC on the original space')
classify_and_test(Xtr, ytr, Xte, yte)

# test the DRO
dro = DistributionalRandomOversampling(rebalance_ratio=0.1)
Xtr, ytr = dro.fit_transform(Xtr, ytr, train_nwords)
Xte = dro.transform(Xte, test_nwords, samples=5)

positives = ytr.sum()
nD = len(ytr)
print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')
print(Xtr.shape)
print(Xte.shape)

print('LinearSVC on the oversampled space')
classify_and_test(Xtr, ytr, Xte, yte, average=5)

