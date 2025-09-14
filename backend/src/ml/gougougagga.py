import torch
data = torch.load("./data/preprocessed_data/processed_data.pt")
from tqdm import tqdm 
train_pairs, val_pairs, test_pairs = data["train"], data["val"], data["test"]
fr_word2id = data["fr_word2id"]
fr_id2word = data["fr_id2word"]
en_word2id = data["en_word2id"]


print(train_pairs[0][1])

word_train,word_val,word_test = 0, 0, 0
used_train, used_val, used_test = [], [], []
used_train_p, used_val_p, used_test_p = [], [], []
dtr,dva,dte = 0, 0 ,0

for pairs in tqdm(train_pairs):
    if pairs[1] not in used_train_p:
        used_train_p.append(pairs[1])
        for index in pairs[1]:
            if index not in used_train:
                used_train.append(index)
                word_train += 1
    else:
        dtr += 1



for pairs in tqdm(test_pairs):
    if pairs[1] not in used_test_p:
        used_test_p.append(pairs[1])
        for index in pairs[1]:
            if index not in used_test:
                used_test.append(index)
                word_test += 1
    else:
        dte += 1


for pairs in tqdm(val_pairs):
    if pairs[1] not in used_val_p:
        used_val_p.append(pairs[1])
        for index in pairs[1]:
            if index not in used_val:
                used_val.append(index)
                word_val += 1
    else:
        dva += 1

total_word = len(fr_id2word)

print(f" {100 * word_train/total_word} train ")
print(f" {100 * word_test/total_word} test ")
print(f" {100 * word_val/total_word} val ")

print(f"{dtr} doublon train")
print(f"{dte} doublon test")
print(f"{dva} doublon val")