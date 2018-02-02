from collections import Counter
import collections
from operator import itemgetter


def eg1(file_name):
    wordstring = '1 it was the best of times it was the worst of times '
    wordstring += 'it was the age of wisdom it was the age of foolishness'

    wordlist = wordstring[1:].split() # don't count label 1

    wordfreq = [wordlist.count(w) for w in wordlist] # a list comprehension
    pairs = sorted(set(zip(wordlist, wordfreq)))

    print("String\n" + wordstring +"\n")
    print("List\n" + str(wordlist) + "\n")
    print("Frequencies\n" + str(wordfreq) + "\n")
    print("Pairs\n" + str(pairs))

def eg2(file_name):
    wordstring = '1 it was the best of times it was the worst of times '
    wordstring += 'it was the age of wisdom it was the age of foolishness'

    wordlist = wordstring[1:].split() # don't count label 1

    wordfreq = [wordlist.count(w) for w in wordlist]
    pairs = sorted(set(zip(wordlist, wordfreq)))
    pairs = [(i,j) for i,j in pairs if j>1]
    
    pairs = sorted(pairs, key=lambda word: word[0], reverse=True)

    print(pairs)

def eg3():
    student_tuples = [
        ('john', 'A', 15),
        ('jane', 'B', 12),
        ('dave', 'B', 10),
    ]
    
    a = sorted(student_tuples, key=lambda student: student[0], reverse=True)   # sort by name
    # a = sorted(student_tuples, key=lambda student: student[2], reverse=True)   # sort by age
    print( a)


def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist,wordfreq))

# Sort a dictionary of word-frequency pairs in
# order of descending frequency.

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux


def read_example(file_name):
    wordcount = collections.Counter()
    with open("data2.txt") as f:
        for line in f:
            wordcount.update(line[1:].split())

    pairs = [(i,j) for i,j in wordcount.items() if j>1]
    pairs = sorted(pairs, key=lambda word: word[0], reverse=False)
    
    print(pairs)


def main():
    """Run main function."""
    file_name = 'data2.txt'
    read_example(file_name)
    # eg2(file_name)
    # eg3()

if __name__ == "__main__":
    main()
