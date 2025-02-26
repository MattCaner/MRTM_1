import numpy as np
import math
import random

n = np.arange(2000)

ff = open('cyclic.csv','w')

for i in n:
    ff.write(str(i) + ',' + str(2*math.sin(i)+3.*math.sin(3.*i+1)+5.*math.sin(5.*i+2)+random.random()-0.5) +'\n')

ff.close()