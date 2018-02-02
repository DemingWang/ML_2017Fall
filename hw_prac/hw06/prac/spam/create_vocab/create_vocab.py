import collections
import numpy as np

def create_vocab(fdata,min_freq,fvocab):

    # count the words and frequencies
    wordcount = collections.Counter()
    with open(fdata) as fi:
        for line in fi:
            wordcount.update(set(line[2:].split()))

    pairs = [(w,f) for w,f in wordcount.items() if f>=min_freq ]

    # do not include stopwords
    fstopwords = '../data/stopwords.txt'
    stopwords = np.loadtxt(fstopwords,dtype='str')
    pairs = [(w,f) for w,f in wordcount.items() if f>=min_freq if w not in stopwords]

    # sort alphabetically
    # pairs = sorted(pairs, key=lambda word: word[0], reverse=0)

    # sort by number of occurrence
    pairs = sorted(pairs, key=lambda word: word[1], reverse=1)

    print("len(vocab) = {}".format(len(pairs)))
    with open(fvocab,'w') as fo:
        for i in range(len(pairs)):
            # fo.write("{} {}\n".format(i+1,pairs[i][0]))

            # write index token freq
            fo.write("{} {} {}\n".format(i+1,pairs[i][0], pairs[i][1]))
            
def main():
    """Run main function."""
    # fdata, min_freq, fvocab = '../data/data_01.txt', 30, '../data/vocab_01.txt'
    fdata, min_freq, fvocab = '../data/data_m1.txt', 30, '../data/vocab_m1.txt'
    create_vocab(fdata,min_freq,fvocab)

if __name__ == "__main__":
    main()
