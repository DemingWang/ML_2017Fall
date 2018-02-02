import numpy as np
def create_vocab(fdata,min_freq, fvocab):        
    vocab = {}
    words = []
 
    for line  in open(fdata):
        l = list(set(line.lstrip('0123456789.- ').split()))  
        for w in l:
            words.append(w)
    
    # create dictionay
    for w in words:
        if words.count(w) >= min_freq:
            vocab[w] = words.count(w)
    
    # sorted list from the dictionary
    sorted_lst = sorted(vocab.items(), key=lambda value: value[0])
    
    # create vocab
    with open(fvocab,'w') as fo:
        for tpl in enumerate(sorted_lst):
            print(tpl[0]+1,tpl[1][0], file=fo)

def main():
    """Run main function."""
    vocab = create_vocab('../data/data.txt',2,'../data/vocab.txt')


if __name__ == "__main__":
    main()
