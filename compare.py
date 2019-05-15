
A = open('MultiEvoMunkres.py','r')
B = open('SparseMultiEvoMunkres.py','r')

c = 0

for i,(a,b) in enumerate(zip(A.readlines(),B.readlines())):
    if i > 48 and str(a) != str(b):
        print('Line {}, :'.format(i))
        print('/t',a)
        print('/t',b)
        c += 1
    if c> 10:
        break