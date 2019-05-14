import numpy as np
from scipy.special import binom

class ptri:
    def __init__(self):
        self.tri = {0:{0:1.,1:1.},1:{0:1.}}

    def get(self,x,y):
        if x == 0 or y == 0:
            if x not in self.tri:
                self.tri[x] = dict()
            if y not in self.tri[x]:
                self.tri[x][y] = 1
            return 1
        if (x - 1) not in self.tri or y not in self.tri[(x - 1)]:
            self.get((x - 1), y)
        if x not in self.tri:
            self.tri[x] = dict()
        if (y-1) not in self.tri[x]:
            self.get(x,(y-1))

        if y not in self.tri[x]:
            self.tri[x][y] = self.tri[(x - 1)][y] + self.tri[x][(y - 1)]

        return self.tri[x][y]


def rules(x,y,v=False):
    if x == y:
        return False
    if binom(x-1,y)%2:
        return True
    return False

def pow2(x):
    if x <= 0:
        return False
    return not (x & (x-1))


if __name__ == '__main__':
    n = 8
    mat = np.zeros((n,n))
    pt = ptri()

    for a in range(n-1):
        for b in range(a+1):
            x = (a-b)
            y = b
            mat[a+1][b] = pt.get(x,y)%2
            mat[b][a+1] = mat[a+1][b]
    mat = np.multiply(mat,1-np.eye(n))
    # print(mat)
    for i in range(n):
        adj = np.where(mat[i,:]>0)[0].tolist()
        print('For row {}:'.format(i),adj)


    tmat = np.zeros_like(mat)
    for a in range(n):
        for b in range(a):
            tmat[a][b] = mat[a,b] - int(rules(a,b))
            tmat[b][a] = tmat[a][b]
            if abs(tmat[a][b]):
                print('T',a,b,tmat[a][b])
                rules(a,b,v=True)
    print(mat)
    print(tmat)

    # print(tmat)

