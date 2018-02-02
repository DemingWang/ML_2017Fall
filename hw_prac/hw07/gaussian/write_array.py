import numpy as np

a = np.array([ [0,1,2],
              [10,20,30]])

np.savetxt('tmp.txt',a,fmt='%g',delimiter=',')
