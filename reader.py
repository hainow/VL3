'''
This file contains the class for reading text files.
Please fill up the functions oneHot and convertHot.
'''
import numpy as np
import random
import pdb

class textDataLoader(object):
    def __init__(self, datapath):
        self.nFeats = 128
        print 'Loading Data'
        self.D = open(datapath, 'r').read()
        self.nChars = len(self.D)
        self.N = [ord(c) for c in self.D]
        self.D_oneHot = np.array([self.oneHot(n) for n in self.N])

    def oneHot(self, character):
        '''
        In this function you need to output a one hot encoding of the ASCII character.
        '''
        one_hot = np.zeros((self.nFeats, ))
        one_hot[character] = 1
        return one_hot

    def convertHot(self, string_l):
        '''
        In this function, you will need to write a piece of code that converts a string
        to a numpy array of one hot representation.
        '''
        l = len(string_l)
        one_hot_string = np.zeros((l, self.nFeats))
        for i in range(l):
            one_hot_string[i] = self.oneHot(ord(string_l[i]))
        return one_hot_string

    def getBatch(self, batch_size, max_length):
        input_b = np.zeros([batch_size, max_length, self.nFeats])
        output_b = np.zeros([batch_size, max_length, self.nFeats])
        for i in range(batch_size):
            r = random.randint(0, self.nChars - 2 - max_length)
            # pdb.set_trace()
            input_b[i] = self.D_oneHot[r:r + max_length]
            output_b[i] = self.D_oneHot[r + 1:r + 1 + max_length]
        return input_b, output_b


def main():
    x = textDataLoader('dataset_small.txt')
    # pdb.set_trace()

    s = 'You are all resolved rather to die'
    print(x.convertHot(s)[1])


if __name__ == "__main__":
    main()