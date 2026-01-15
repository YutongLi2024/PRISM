import gzip
from collections import defaultdict
from datetime import datetime
import os
import json
from tqdm import tqdm
import pandas as pd
true = True
false = False

print("Current Working Directory:", os.getcwd())


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

DATASET = "Sports"
dataname = os.path.join(DATASET, f'reviews_{DATASET}_and_Outdoors_5.json.gz')
print(dataname)  # Beauty/reviews_Beauty_5.json.gz

if not os.path.isdir('./'+DATASET):
    os.mkdir('./'+DATASET)
train_file = './'+DATASET+'/train.txt'
valid_file = './'+DATASET+'/valid.txt'
test_file = './'+DATASET+'/test.txt'
imap_file = './'+DATASET+'/imap.json'
umap_file = './'+DATASET+'/umap.json'

data_file = './'+DATASET+f'/{DATASET}.txt'


for one_interaction in tqdm(parse(dataname)):
    rev = one_interaction['reviewerID']
    asin = one_interaction['asin']
    time = float(one_interaction['unixReviewTime'])
    countU[rev] += 1
    countP[asin] += 1

usermap = dict()
usernum = 1
itemmap = dict()
itemnum = 1
User = dict()


for one_interaction in tqdm(parse(dataname)):
    rev = one_interaction['reviewerID']
    asin = one_interaction['asin']
    time = float(one_interaction['unixReviewTime'])
    if countU[rev] < 5 or countP[asin] < 5:
        continue 

    if rev in usermap:
        userid = usermap[rev]
    else:
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
        usernum += 1
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemid = itemnum
        itemmap[asin] = itemid
        itemnum += 1
    User[userid].append([itemid, time])


with open(imap_file, 'w') as f:
    json.dump(itemmap, f)

with open(umap_file, 'w') as f:
    json.dump(usermap, f)

for userid in User.keys():
    # sort reviews in User according to time
    User[userid].sort(key=lambda x: x[1])


user_train = {}
user_valid = {}
user_test = {}
for user in User:
    nfeedback = len(User[user])
    if nfeedback < 3:
        user_train[user] = User[user]
        user_valid[user] = []
        user_test[user] = []
    else:
        user_train[user] = User[user][:-2]
        user_valid[user] = []
        user_valid[user].append(User[user][-2])
        user_test[user] = []
        user_test[user].append(User[user][-1])


print(usernum, itemnum)


def write_file_v2(data, dfile):
    with open(dfile, 'w') as f:
        for u, ilist in sorted(data.items()):
            f.write(str(u))
            for i, t in ilist:
                f.write(' '+ str(i))
            f.write("\n")


write_file_v2(User, data_file)

num_instances = sum([len(ilist) for _, ilist in User.items()])
print('total user: ', len(User))
print('total instances: ', num_instances)
print('avg length: ', num_instances / len(User))
print('total items: ', itemnum)
print('density: ', num_instances / (len(User) * itemnum))
print('valid #users: ', len(user_valid))
numvalid_instances = sum([len(ilist) for _, ilist in user_valid.items()])
print('valid instances: ', numvalid_instances)
numtest_instances = sum([len(ilist) for _, ilist in user_test.items()])
print('test #users: ', len(user_test))
print('test instances: ', numtest_instances)
