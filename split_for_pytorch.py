import gc
import os
idx = 0
idx_dir = 0
os.makedirs('data/train_0')
for j in range(5):
    with open('train1_{}.txt'.format(j), 'r') as f:
        for line in f:
            train_part = open('data/train_{}/train_{}.txt'.format(idx_dir, idx), 'w')
            train_part.write(line)
            train_part.close()
            idx += 1
            if idx % 10000 == 0:
                idx_dir += 1
                os.makedirs('data/train_{}'.format(idx_dir))
            print(idx)