import numpy as np
from collections import Counter
import random
import sys

np.random.seed(12345)

f = open('spam.txt','r')
raw = f.readlines()
f.close()

spam = list()
for row in raw:
    spam.append(row[:-2].split(" "))
    
f = open('ham.txt','r')
raw = f.readlines()
f.close()

ham = list()
for row in raw:
    ham.append(row[:-2].split(" "))
    
class LogisticRegression(object):
    
    def __init__(self, positives,negatives,iterations=10,alpha=0.1):
        
        # create vocabulary (real world use case would add a few million
        # other terms as well from a big internet scrape)
        cnts = Counter()
        for email in (positives+negatives):
            for word in email:
                cnts[word] += 1
        
        # convert to lookup table
        vocab = list(cnts.keys())
        self.word2index = {}
        for i,word in enumerate(vocab):
            self.word2index[word] = i
    
        # initialize decrypted weights
        self.weights = (np.random.rand(len(vocab)) - 0.5) * 0.1
        
        # train model on unencrypted information
        self.train(positives,negatives,iterations=iterations,alpha=alpha)
    
    def train(self,positives,negatives,iterations=10,alpha=0.1):
        
        for iter in range(iterations):
            error = 0
            n = 0
            for i in range(max(len(positives),len(negatives))):

                error += np.abs(self.learn(positives[i % len(positives)],1,alpha))
                error += np.abs(self.learn(negatives[i % len(negatives)],0,alpha))
                n += 2

            print("Iter:" + str(iter) + " Loss:" + str(error / float(n)))

    
    def softmax(self,x):
        return 1/(1+np.exp(-x))

    def predict(self,email):
        pred = 0
        for word in email:
            pred += self.weights[self.word2index[word]]
        pred = self.softmax(pred)
        return pred

    def learn(self,email,target,alpha):
        pred = self.predict(email)
        delta = (pred - target)# * pred * (1 - pred)
        for word in email:
            self.weights[self.word2index[word]] -= delta * alpha
        return delta
    
model = LogisticRegression(spam[0:-100],ham[0:-100],iterations=3)

# evaluate on holdout set

fp = 0
tn = 0
tp = 0
fn = 0

for i,h in enumerate(ham[-100:]):
    pred = model.predict(h)

    if(pred < 0.5):
        tn += 1
    else:
        fp += 1
        
    if(i % 10 == 0):
        sys.stdout.write('\rI:'+str(tn+tp+fn+fp) + " % Correct:" + str(100*tn/float(tn+fp))[0:6])

for i,h in enumerate(spam[-100:]):
    pred = model.predict(h)

    if(pred >= 0.5):
        tp += 1
    else:
        fn += 1

    if(i % 10 == 0):
        sys.stdout.write('\rI:'+str(tn+tp+fn+fp) + " % Correct:" + str(100*(tn+tp)/float(tn+tp+fn+fp))[0:6])
sys.stdout.write('\rI:'+str(tn+tp+fn+fp) + " Correct: %" + str(100*(tn+tp)/float(tn+tp+fn+fp))[0:6])

print("\nTest Accuracy: %" + str(100*(tn+tp)/float(tn+tp+fn+fp))[0:6])
print("False Positives: %" + str(100*fp/float(tp+fp))[0:4] + "    <- privacy violation level out of 100.0%")
print("False Negatives: %" + str(100*fn/float(tn+fn))[0:4] + "   <- security risk level out of 100.0%") 
