

def word_count(data,min_freq,vocab):
    counts = dict()
    
    with open(data) as fi:
        for line in fi:
            words = line[1:].split()
            for word in words:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
                    
        pairs = [(w,f) for w,f in counts.items() if f>=min_freq ]
        
        pairs = sorted(pairs, key=lambda word: word[0], reverse=False)
        
        # print("len(pairs) = {}".format(len(pairs)))
        with open(vocab,'w') as fo:
            for i in range(len(pairs)):
                # fo.write("{} {}\n".format(i+1,pairs[i][0]))
                fo.write("{} {} {}\n".format(i+1,pairs[i][0], pairs[i][1]))

    return counts

def main():
    """Run main function."""
    # s = "apple banana egg cow egg aa apple"
    # print(word_count(s))
    
    data = 'data.txt'
    vocab = 'vocab.txt'
    # data = '../spam/spam_train.txt'
    min_freq = 2
    c = word_count(data,min_freq,vocab)
    # print(c)
    
    # checking
    """
    grep -o -c aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ../spam/spam_train.txt   # 1
    
    # grep gives occurerence of word_part
    # it counts b twice in b and ball.
    # this means single line has 141 occurrence of that string.
    
    # my implementation gives
    3238 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 141
    """

    

if __name__ == "__main__":
    main()
    
