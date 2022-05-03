import pickle, sys
sys.path.append('./parchoice')
sys.path.append('./parchoice/style_transfer')
sys.path.append('./transformer')
from surrogate_classifier import surrogate_kwargs
from style_transformation import TfidfVectorizer, MLPSurrogate, LogisticRegressionSurrogate
from os import path


with open('elon_test.txt', 'r') as f:
    src = f.readlines()

with open('trump_test.txt', 'r') as f:
    tgt = f.readlines()

with open('data/elon_train.txt', 'r') as f:
    src_train = f.readlines()

with open('data/trump_train.txt', 'r') as f:
    tgt_train = f.readlines()

with open('parchoice_only_out_src.txt', 'r') as f:
    pc_src = f.readlines()

with open('parchoice_only_out_tgt.txt', 'r') as f:
    pc_tgt = f.readlines()

with open('transformer_only_out_src.txt', 'r') as f:
    tr_src = f.readlines()

with open('transformer_only_out_tgt.txt', 'r') as f:
    tr_tgt = f.readlines()

with open('serial_parchoice_transformer_out_src.txt', 'r') as f:
    pc_tr_src = f.readlines()

with open('serial_parchoice_transformer_out_tgt.txt', 'r') as f:
    pc_tr_tgt = f.readlines()

with open('serial_transformer_parchoice_out_src.txt', 'r') as f:
    tr_pc_src = f.readlines()

with open('serial_transformer_parchoice_out_tgt.txt', 'r') as f:
    tr_pc_tgt = f.readlines()

with open('hybrid_transformer_parchoice_out_src.txt', 'r') as f:
    hy_src = f.readlines()

with open('hybrid_transformer_parchoice_out_tgt.txt', 'r') as f:
    hy_tgt = f.readlines()

clf = None

if not path.exists('models/LR_trump_elon_clf.pkl'):
    surrogate_class = LogisticRegressionSurrogate
    surrogate_vectorizer = TfidfVectorizer
    surrogate_corpus = src_train + tgt_train
    surrogate_corpus_labels = [0 for _ in src_train] + [1 for _ in tgt_train]
    print('\nTraining classifier...', end=' ')
    clf = surrogate_class(surrogate_vectorizer, surrogate_kwargs(surrogate_vectorizer, 'word', (1,1), 10000), 1).fit(surrogate_corpus, surrogate_corpus_labels)
    print('Done!')
    with open('models/LR_trump_elon_clf.pkl', 'wb') as f:
        pickle.dump(clf, f)
else:
    with open('models/LR_trump_elon_clf.pkl', 'rb') as f:
        clf = pickle.load(f)

if not path.exists('models/LR_trump_elon_clf_2.pkl'):
    surrogate_class = LogisticRegressionSurrogate
    surrogate_vectorizer = TfidfVectorizer
    surrogate_corpus = tgt_train + src_train 
    surrogate_corpus_labels = [0 for _ in tgt_train] + [1 for _ in src_train]
    print('\nTraining classifier...', end=' ')
    clf2 = surrogate_class(surrogate_vectorizer, surrogate_kwargs(surrogate_vectorizer, 'word', (1,1), 10000), 1).fit(surrogate_corpus, surrogate_corpus_labels)
    print('Done!')
    with open('models/LR_trump_elon_clf_2.pkl', 'wb') as f:
        pickle.dump(clf2, f)
else:
    with open('models/LR_trump_elon_clf_2.pkl', 'rb') as f:
        clf2 = pickle.load(f)

acc = clf.accuracy(src, [1 for i in range(len(src))])
print(f"Accuracy of classifier on source: {acc}")
acc = clf2.accuracy(tgt, [1 for i in range(len(tgt))])
print(f"Accuracy of classifier on target: {acc}")
acc = clf.accuracy(pc_src, [1 for i in range(len(pc_src))])
print(f"Accuracy of classifier on parchoice only source: {acc}")
acc = clf2.accuracy(pc_tgt, [1 for i in range(len(pc_tgt))])
print(f"Accuracy of classifier on parchoice only target: {acc}")
acc = clf.accuracy(tr_src, [1 for i in range(len(tr_src))])
print(f"Accuracy of classifier on transformer only source: {acc}")
acc = clf2.accuracy(tr_tgt, [1 for i in range(len(tr_tgt))])
print(f"Accuracy of classifier on transformer only target: {acc}")
acc = clf.accuracy(pc_tr_src, [1 for i in range(len(pc_tr_src))])
print(f"Accuracy of classifier on serial parchoice transformer source: {acc}")
acc = clf2.accuracy(pc_tr_tgt, [1 for i in range(len(pc_tr_tgt))])
print(f"Accuracy of classifier on serial parchoice transformer target: {acc}")
acc = clf.accuracy(tr_pc_src, [1 for i in range(len(tr_pc_src))])
print(f"Accuracy of classifier on serial transformer parchoice source: {acc}")
acc = clf2.accuracy(tr_pc_tgt, [1 for i in range(len(tr_pc_tgt))])
print(f"Accuracy of classifier on serial transformer parchoice target: {acc}")
acc = clf.accuracy(hy_src, [1 for i in range(len(hy_src))])
print(f"Accuracy of classifier on hybrid transformer parchoice source: {acc}")
acc = clf2.accuracy(hy_tgt, [1 for i in range(len(hy_tgt))])
print(f"Accuracy of classifier on hybrid transformer parchoice target: {acc}")

print("")
print("Using MLP Surrogate Model:")

if not path.exists('models/MLP_trump_elon.pkl'):
    surrogate_class = MLPSurrogate
    surrogate_vectorizer = TfidfVectorizer
    surrogate_corpus = src_train + tgt_train
    surrogate_corpus_labels = [0 for _ in src_train] + [1 for _ in tgt_train]
    print('\nTraining classifier...', end=' ')
    clf = surrogate_class(surrogate_vectorizer, surrogate_kwargs(surrogate_vectorizer, 'word', (1,1), 10000), 1).fit(surrogate_corpus, surrogate_corpus_labels)
    print('Done!')
    with open('models/MLP_trump_elon.pkl', 'wb') as f:
        pickle.dump(clf, f)
else:
    with open('models/MLP_trump_elon.pkl', 'rb') as f:
        clf = pickle.load(f)

if not path.exists('models/MLP_trump_elon_2.pkl'):
    surrogate_class = MLPSurrogate
    surrogate_vectorizer = TfidfVectorizer
    surrogate_corpus = tgt_train + src_train 
    surrogate_corpus_labels = [0 for _ in tgt_train] + [1 for _ in src_train]
    print('\nTraining classifier...', end=' ')
    clf2 = surrogate_class(surrogate_vectorizer, surrogate_kwargs(surrogate_vectorizer, 'word', (1,1), 10000), 1).fit(surrogate_corpus, surrogate_corpus_labels)
    print('Done!')
    with open('models/MLP_trump_elon_2.pkl', 'wb') as f:
        pickle.dump(clf2, f)
else:
    with open('models/MLP_trump_elon_2.pkl', 'rb') as f:
        clf2 = pickle.load(f)

acc = clf.accuracy(src, [1 for i in range(len(src))])
print(f"Accuracy of classifier on source: {acc}")
acc = clf2.accuracy(tgt, [1 for i in range(len(tgt))])
print(f"Accuracy of classifier on target: {acc}")
acc = clf.accuracy(pc_src, [1 for i in range(len(pc_src))])
print(f"Accuracy of classifier on parchoice only source: {acc}")
acc = clf2.accuracy(pc_tgt, [1 for i in range(len(pc_tgt))])
print(f"Accuracy of classifier on parchoice only target: {acc}")
acc = clf.accuracy(tr_src, [1 for i in range(len(tr_src))])
print(f"Accuracy of classifier on transformer only source: {acc}")
acc = clf2.accuracy(tr_tgt, [1 for i in range(len(tr_tgt))])
print(f"Accuracy of classifier on transformer only target: {acc}")
acc = clf.accuracy(pc_tr_src, [1 for i in range(len(pc_tr_src))])
print(f"Accuracy of classifier on serial parchoice transformer source: {acc}")
acc = clf2.accuracy(pc_tr_tgt, [1 for i in range(len(pc_tr_tgt))])
print(f"Accuracy of classifier on serial parchoice transformer target: {acc}")
acc = clf.accuracy(tr_pc_src, [1 for i in range(len(tr_pc_src))])
print(f"Accuracy of classifier on serial transformer parchoice source: {acc}")
acc = clf2.accuracy(tr_pc_tgt, [1 for i in range(len(tr_pc_tgt))])
print(f"Accuracy of classifier on serial transformer parchoice target: {acc}")
acc = clf.accuracy(hy_src, [1 for i in range(len(hy_src))])
print(f"Accuracy of classifier on hybrid transformer parchoice source: {acc}")
acc = clf2.accuracy(hy_tgt, [1 for i in range(len(hy_tgt))])
print(f"Accuracy of classifier on hybrid transformer parchoice target: {acc}")