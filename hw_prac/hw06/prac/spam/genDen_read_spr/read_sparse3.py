import numpy as np

def eg1():
    
    pat = r'([0-1])\s+([0-9]):(1)\s+([0-9]):(1)(?:\s+([0-9]):(1))?'

    data = np.fromregex('sparse.txt', pat, dtype='str')
    
    print("data = {}".format(data))
    print("data.shape = {}".format(data.shape))

def main():
    """Run main function."""
    eg1()

if __name__ == "__main__":
    main()
