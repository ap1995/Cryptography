import phe as paillier
import numpy as np
from collections import Counter
import sys
import time

np.random.seed(12345)

print("Generating paillier keypair")
pubkey, prikey = paillier.generate_paillier_keypair(n_length=64)


class HomomorphicLogisticRegression(object):

    def __init__(self, iterations=10, alpha=0.1):

        self.iterations = iterations
        self.alpha = alpha
        self.encrypted = False
        self.maxweight = 10
        self.total_time = 0.0

        def create_dataset(file_path):
            f = open(file_path, 'r', encoding='ISO-8859-1')
            raw = f.readlines()
            f.close()
            data = list()
            for row in raw:
                data.append(row[:-2].split(" "))
            return data

        spam_dataset = create_dataset('spam.txt')
        ham_dataset = create_dataset('ham.txt')
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
            self.total_time += time_end - time_start

    def softmax(self, x):
        return 1 / (1 + np.exp(-x))

    def encrypt(self, pubkey, scaling_factor=1000):
        if (not self.encrypted):
            self.pubkey = pubkey
            self.scaling_factor = float(scaling_factor)
            self.encrypted_weights = list()

            for weight in model.weights:
                self.encrypted_weights.append(
                    self.pubkey.encrypt(int(min(weight, self.maxweight) * self.scaling_factor)))

            self.encrypted = True
            self.weights = None

        return self

    def predict(self, email):
        if (self.encrypted):
            return self.encrypted_predict(email)
        else:
            return self.unencrypted_predict(email)

    def encrypted_predict(self, email):
        pred = self.pubkey.encrypt(0)
        for word in email:
            pred += self.encrypted_weights[self.word_freq[word]]
        return pred

    def unencrypted_predict(self, email):
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

    def get_average_time(self):
        return self.total_time / self.iterations


model = HomomorphicLogisticRegression(iterations=10)

# create vocabulary dictionary, weights to be used for training
model.create_word_count()

# train model on unencrypted information
model.train()

encrypt_time_start = time.time()
encrypted_model = model.encrypt(pubkey)
encrypt_time_end = time.time()
# generate encrypted predictions. Then decrypt them and evaluate.

fp = 0
tn = 0
tp = 0
fn = 0
total_time = 0.0

for i, h in enumerate(model.test_positives):
    time_start = time.time()
    encrypted_pred = encrypted_model.predict(h)
    time_end = time.time()
    total_time += time_end - time_start

    try:
        pred = prikey.decrypt(encrypted_pred) / encrypted_model.scaling_factor
        if (pred < 0):
            tn += 1
        else:
            fp += 1
    except:
        print("overflow")

for i, h in enumerate(model.test_negatives):
    time_start = time.time()
    encrypted_pred = encrypted_model.predict(h)
    time_end = time.time()
    total_time += time_end - time_start

    try:
        pred = prikey.decrypt(encrypted_pred) / encrypted_model.scaling_factor
        if (pred > 0):
            tp += 1
        else:
            fn += 1
    except:
        print("overflow")

print("Average training time {:0.2f}".format(model.get_average_time()))
print("Time taken for encryption {:0.2f}".format(encrypt_time_end - encrypt_time_start))
print("Average prediction time(micro s) {:0.2f}".format(total_time / 200 * 10 ** 6))
print("Encrypted Accuracy: %" + str(100 * (tn + tp) / float(tn + tp + fn + fp))[0:6])
print("False Positives: %" + str(100 * fp / float(tp + fp))[0:4] + "    <- privacy violation level")
print("False Negatives: %" + str(100 * fn / float(tn + fn))[0:4] + "   <- security risk level")
