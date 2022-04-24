import argparse, sys, torch, random, pickle, time
sys.path.append('./parchoice')
sys.path.append('./parchoice/style_transfer')
sys.path.append('./transformer')
from os import path, remove
from main_parchoice import parchoice
from inference import inference
from context_preservation import preserve_context

with open('elon_test.txt', 'r') as f:
    src = f.readlines()

with open('trump_test.txt', 'r') as f:
    tgt = f.readlines()

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

with open('hybrid_transformer_parchoice_out_src.txt', 'r') as f:
    hy_tgt = f.readlines()

clf = None

with open('models/LR_trump_elon_clf.pkl', 'rb') as f:
    clf = pickle.load(f)

acc = clf.accuracy(src, [0 for i in range(len(src))])
print(f"Accuracy of classifier on source: {acc}")
acc = clf.accuracy(tgt, [1 for i in range(len(tgt))])
print(f"Accuracy of classifier on target: {acc}")
acc = clf.accuracy(pc_src, [0 for i in range(len(pc_src))])
print(f"Accuracy of classifier on parchoice only source: {acc}")
acc = clf.accuracy(pc_tgt, [1 for i in range(len(pc_tgt))])
print(f"Accuracy of classifier on parchoice only target: {acc}")
acc = clf.accuracy(tr_src, [0 for i in range(len(tr_src))])
print(f"Accuracy of classifier on transformer only source: {acc}")
acc = clf.accuracy(tr_tgt, [1 for i in range(len(tr_tgt))])
print(f"Accuracy of classifier on transformer only target: {acc}")
acc = clf.accuracy(pc_tr_src, [0 for i in range(len(pc_tr_src))])
print(f"Accuracy of classifier on serial parchoice transformer source: {acc}")
acc = clf.accuracy(pc_tr_tgt, [1 for i in range(len(pc_tr_tgt))])
print(f"Accuracy of classifier on serial parchoice transformer target: {acc}")
acc = clf.accuracy(tr_pc_src, [0 for i in range(len(tr_pc_src))])
print(f"Accuracy of classifier on serial transformer parchoice source: {acc}")
acc = clf.accuracy(tr_pc_tgt, [1 for i in range(len(tr_pc_tgt))])
print(f"Accuracy of classifier on serial transformer parchoice target: {acc}")
acc = clf.accuracy(hy_src, [0 for i in range(len(hy_src))])
print(f"Accuracy of classifier on hybrid transformer parchoice source: {acc}")
acc = clf.accuracy(hy_tgt, [1 for i in range(len(hy_tgt))])
print(f"Accuracy of classifier on hybrid transformer parchoice target: {acc}")