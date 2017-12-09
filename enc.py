import phe as paillier
import math
import numpy as np
from collections import Counter
import random
import sys

np.random.seed(12345)

print("Generating paillier keypair")
pubkey, prikey = paillier.generate_paillier_keypair(n_length=64)

print("Importing dataset from disk...")
f = open('spam.txt','r',encoding='ISO-8859-1')
raw = f.readlines()
f.close()

spam = list()
for row in raw:
    spam.append(row[:-2].split(" "))
    
f = open('ham.txt','r',encoding='ISO-8859-1')
raw = f.readlines()
f.close()

ham = list()
for row in raw:
    ham.append(row[:-2].split(" "))
    
class HomomorphicLogisticRegression(object):
    
    def __init__(self, positives,negatives,iterations=10,alpha=0.1):
        
        self.encrypted=False
        self.maxweight=10
        
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

    def encrypt(self,pubkey,scaling_factor=1000):
        if(not self.encrypted):
            self.pubkey = pubkey
            self.scaling_factor = float(scaling_factor)
            self.encrypted_weights = list()

            for weight in model.weights:
                self.encrypted_weights.append(self.pubkey.encrypt(int(min(weight,self.maxweight) * self.scaling_factor)))

            self.encrypted = True            
            self.weights = None

            
        return self

    def predict(self,email):
        if(self.encrypted):
            return self.encrypted_predict(email)
        else:
            return self.unencrypted_predict(email)
    
    def encrypted_predict(self,email):
        pred = self.pubkey.encrypt(0)
        for word in email:
            pred += self.encrypted_weights[self.word2index[word]]
        return pred
    
    def unencrypted_predict(self,email):
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
    
model = HomomorphicLogisticRegression(spam[0:-1000],ham[0:-1000],iterations=10)

encrypted_model = model.encrypt(pubkey)

# generate encrypted predictions. Then decrypt them and evaluate.

fp = 0
tn = 0
tp = 0
fn = 0

for i,h in enumerate(ham[-1000:]):
    encrypted_pred = encrypted_model.predict(h)
    try:
        pred = prikey.decrypt(encrypted_pred) / encrypted_model.scaling_factor
        if(pred < 0):
            tn += 1
        else:
            fp += 1
    except:
        print("overflow")

    if(i % 10 == 0):
        sys.stdout.write('\r I:'+str(tn+tp+fn+fp) + " % Correct:" + str(100*tn/float(tn+fp))[0:6])

for i,h in enumerate(spam[-1000:]):
    encrypted_pred = encrypted_model.predict(h)
    try:
        pred = prikey.decrypt(encrypted_pred) / encrypted_model.scaling_factor
        if(pred > 0):
            tp += 1
        else:
            fn += 1
    except:
        print("overflow")

    if(i % 10 == 0):
        sys.stdout.write('\r I:'+str(tn+tp+fn+fp) + " % Correct:" + str(100*(tn+tp)/float(tn+tp+fn+fp))[0:6])
sys.stdout.write('\r I:'+str(tn+tp+fn+fp) + " % Correct:" + str(100*(tn+tp)/float(tn+tp+fn+fp))[0:6])

print("\n Encrypted Accuracy: %" + str(100*(tn+tp)/float(tn+tp+fn+fp))[0:6])
print("False Positives: %" + str(100*fp/float(tp+fp))[0:4] + "    <- privacy violation level")
print("False Negatives: %" + str(100*fn/float(tn+fn))[0:4] + "   <- security risk level") 