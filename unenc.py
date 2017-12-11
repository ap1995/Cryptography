import numpy as np
from collections import Counter
import sys
import time

np.random.seed(12345)


class LogisticRegression(object):

    def __init__(self, iterations=10, alpha=0.1):

        self.iterations = iterations
        self.alpha = alpha

        def create_dataset(file_path):
            f = open(file_path, 'r', encoding='ISO-8859-1')
            raw = f.readlines()
            f.close()
            data = list()
            for row in raw:
                data.append(row[:-2].split(" "))
            return data

        spam_dataset = create_dataset('prevData/spam.txt')
        ham_dataset = create_dataset('prevData/ham.txt')
        self.positives = spam_dataset[0:-100]
        self.negatives = ham_dataset[0:-100]
        self.test_positives = ham_dataset[-100:]
        self.test_negatives = spam_dataset[-100:]

    def create_word_count(self):
        cnts = Counter()
        for email in (self.positives + self.negatives):
            for word in email:
                cnts[word] += 1
        # convert to lookup table
        vocab = list(cnts.keys())
        self.word_freq = {}
        for i, word in enumerate(vocab):
            self.word_freq[word] = i
        # initialize decrypted weights
        self.weights = (np.random.rand(len(vocab)) - 0.5) * 0.1

    def train(self):

        for iter in range(self.iterations):
            error = 0
            n = 0
            time_start = time.time()
            for i in range(max(len(self.positives), len(self.negatives))):
                error += np.abs(self.learn(self.positives[i % len(self.positives)], 1, self.alpha))
                error += np.abs(self.learn(self.negatives[i % len(self.negatives)], 0, self.alpha))
                n += 2
            time_end = time.time()
            print("Iter:" + str(iter) + " Loss:" + str(error / float(n)))
            print("Time taken = " + str(time_end - time_start))

    def softmax(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, email):
        pred = 0
        for word in email:
            pred += self.weights[self.word_freq[word]]
        pred = self.softmax(pred)
        return pred

    def learn(self, email, target, alpha):
        pred = self.predict(email)
        delta = (pred - target)  # * pred * (1 - pred)
        for word in email:
            self.weights[self.word_freq[word]] -= delta * alpha
        return delta


model = LogisticRegression(iterations=3)

# create vocabulary dictionary, weights to be used for training
model.create_word_count()

# train model on unencrypted information
model.train()

# evaluate on holdout set

fp = 0
tn = 0
tp = 0
fn = 0

for i, h in enumerate(model.test_positives):
    pred = model.predict(h)

    if (pred < 0.5):
        tn += 1
    else:
        fp += 1

    if (i % 10 == 0):
        sys.stdout.write('\rI:' + str(tn + tp + fn + fp) + " % Correct:" + str(100 * tn / float(tn + fp))[0:6])

for i, h in enumerate(model.test_negatives):
    pred = model.predict(h)

    if (pred >= 0.5):
        tp += 1
    else:
        fn += 1

    if (i % 10 == 0):
        sys.stdout.write(
            '\rI:' + str(tn + tp + fn + fp) + " % Correct:" + str(100 * (tn + tp) / float(tn + tp + fn + fp))[0:6])
sys.stdout.write('\rI:' + str(tn + tp + fn + fp) + " Correct: %" + str(100 * (tn + tp) / float(tn + tp + fn + fp))[0:6])

print("\nTest Accuracy: %" + str(100 * (tn + tp) / float(tn + tp + fn + fp))[0:6])
print("False Positives: %" + str(100 * fp / float(tp + fp))[0:4] + "    <- privacy violation level out of 100.0%")
print("False Negatives: %" + str(100 * fn / float(tn + fn))[0:4] + "   <- security risk level out of 100.0%")
