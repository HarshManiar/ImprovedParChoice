### Have to install pip dependencies
#### Download the glove folder from this link:https://nlp.stanford.edu/projects/glove/


from statistics import mode
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
import os
from time import time
from word_mover_distance import model
from statistics import median, mean


def word_mover_distance(s1,s2,model = model.WordEmbedding(model_fn="glove.6B.300d.txt")):
    stop_words = stopwords.words('english')
    s1 = s1.lower().split()
    s2 = s2.lower().split()
    s1 = [word for word in s1 if word not in stop_words]
    s2 = [word for word in s2 if word not in stop_words]
    return model.wmdistance(s1,s2)




#s1= 'Obama speaks to the media in Chicago'
#s2 = 'The president spoke to the press in Chicago'
#start = time()
#print(word_mover_distance(s1,s2))
#print(time()-start)

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

with open('hybrid_transformer_parchoice_out_tgt.txt', 'r') as f:
  hy_tgt = f.readlines()

pc_src_medians = []
pc_tgt_medians = []
tr_src_medians = []
tr_tgt_medians = []
pc_tr_src_medians = []
pc_tr_tgt_medians = []
tr_pc_src_medians = []
tr_pc_tgt_medians = []
hy_src_medians = []
hy_tgt_medians = []

for i in range(1):
  results = []
  for line1, line2 in zip(src, pc_src):
    results.append(word_mover_distance(line1,line2))
  pc_src_medians.append(median(results))

  results = []
  for line1, line2 in zip(tgt, pc_tgt):
    results.append(word_mover_distance(line1,line2))
  pc_tgt_medians.append(median(results))

  results = []
  for line1, line2 in zip(src, tr_src):
    results.append(word_mover_distance(line1,line2))
  tr_src_medians.append(median(results))

  results = []
  for line1, line2 in zip(tgt, tr_tgt):
    results.append(word_mover_distance(line1,line2))
  tr_tgt_medians.append(median(results))

  results = []
  for line1, line2 in zip(src, pc_tr_src):
    results.append(word_mover_distance(line1,line2))
  pc_tr_src_medians.append(median(results))

  results = []
  for line1, line2 in zip(tgt, pc_tr_tgt):
    results.append(word_mover_distance(line1,line2))
  pc_tr_tgt_medians.append(median(results))

  results = []
  for line1, line2 in zip(src, tr_pc_src):
    results.append(word_mover_distance(line1,line2))
  tr_pc_src_medians.append(median(results))

  results = []
  for line1, line2 in zip(tgt, tr_pc_tgt):
    results.append(word_mover_distance(line1,line2))
  tr_pc_tgt_medians.append(median(results))

  results = []
  for line1, line2 in zip(src, hy_src):
    results.append(word_mover_distance(line1,line2))
  hy_src_medians.append(median(results))

  results = []
  for line1, line2 in zip(tgt, hy_tgt):
    results.append(word_mover_distance(line1,line2))
  hy_tgt_medians.append(median(results))

print(f"ParChoice Only Source: {mean(pc_src_medians)}")
print(f"ParChoice Only Target: {mean(pc_tgt_medians)}")
print(f"Transformer Only Source: {mean(tr_src_medians)}")
print(f"Transformer Only Target: {mean(tr_tgt_medians)}")
print(f"Serial ParChoice Transformer Source: {mean(pc_tr_src_medians)}")
print(f"Serial ParChoice Transformer Target: {mean(pc_tr_tgt_medians)}")
print(f"Serial Transformer ParChoice Source: {mean(tr_pc_src_medians)}")
print(f"Serial Transformer ParChoice Target: {mean(tr_pc_tgt_medians)}")
print(f"Hybrid Transformer ParChoice Source: {mean(hy_src_medians)}")
print(f"Hybrid Transformer ParChoice Target: {mean(hy_tgt_medians)}")

