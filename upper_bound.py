import numpy

sum = 0

for i in range(1000):
    sum += (i+1) / (2 ** (i+1))
    
print(sum)