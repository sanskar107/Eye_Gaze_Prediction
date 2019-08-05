import os
import random

f = os.listdir('all/imgs_cropped')
random.shuffle(f)
split = int(0.85*len(f))
train = f[:split]
test = f[split:]

f1 = open('train.txt', 'w')
f2 = open('test.txt', 'w')

print("train : {}".format(len(train)))
print("test : {}".format(len(test)))

for file in train:
	f1.write(file)
	f1.write('\n')

for file in test:
	f2.write(file)
	f2.write('\n')
