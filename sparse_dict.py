import numpy as np

class sparse_dict():
    def __init__(self,mx=0,my=0):
        self.mx = mx
        self.my = my
        self.shape = (mx,my)
        self.data = dict()

    def fill(self,data,rows,cols):
        for d,r,c in zip(data,rows,cols):
            self.set(r,c,d)

    def set(self,x,y,d):
        if d == 0:
            self.rem(x,y)
        elif x in self.data:
            self.data[x][y] = d
        else:
            self.data[x] = {y:d}

    def get(self,x,y):
        if x in self.data and y in self.data[x]:
            return self.data[x][y]
        else:
            return 0

    def __setitem__(self, key, value):
        x,y = key
        if isinstance(x,slice):
            if x.start is None:
                x_start = 0
            else:
                x_start = x.start
            if x.stop is None:
                x_stop = self.mx
            else:
                x_stop = x.stop
            if x.step is None:
                x_step = 1
            else:
                x_step = x.step
        if isinstance(y,slice):
            if y.start is None:
                y_start = 0
            else:
                y_start = y.start
            if y.stop is None:
                y_stop = self.my
            else:
                y_stop = y.stop
            if y.step is None:
                y_step = 1
            else:
                y_step = y.step
        if not isinstance(x,slice):
            if not isinstance(y,slice):
                self.set(x,y,value)
            else:
                for di, yi in enumerate(range(y_start,y_stop,y_step)):
                    if type(value) is np.ndarray:
                        self.set(x,yi,value[di])
                    elif value != 0:
                        self.set(x, yi, value)
                    else:
                        self.rem(x,y)
        elif not isinstance(y,slice):
            for di, xi in enumerate(range(x_start, x_stop, x_step)):
                if type(value) is np.ndarray:
                    self.set(xi,y,value[di])
                elif value != 0:
                    self.set(xi, y, value)
                else:
                    self.rem(x, y)
        else:
            for di, xi in enumerate(range(x_start, x_stop, x_step)):
                for dj, yi in enumerate(range(y_start,y_stop,y_step)):
                    if type(value) is np.ndarray:
                        self.set(xi,yi,value[di,dj])
                    elif value != 0:
                        self.set(xi, yi, value)
                    else:
                        self.rem(xi,yi)

    def __getitem__(self, item):
        if isinstance(item,sparse_dict):
            out = []
            for x in item.data:
                if x in self.data:
                    for y in item.data[x]:
                        if y in self.data[x]:
                            out.append(self.data[x][y])
                            # out.set(x,y,self.data[x][y])
            return np.array(out)
        x,y = item
        if isinstance(x,slice):
            if x.start is None:
                x_start = 0
            else:
                x_start = x.start
            if x.stop is None:
                x_stop = self.mx
            else:
                x_stop = x.stop
            if x.step is None:
                x_step = 1
            else:
                x_step = x.step
        if isinstance(y,slice):
            if y.start is None:
                y_start = 0
            else:
                y_start = y.start
            if y.stop is None:
                y_stop = self.my
            else:
                y_stop = y.stop
            if y.step is None:
                y_step = 1
            else:
                y_step = y.step
        if not isinstance(x,slice):
            if not isinstance(y,slice):
                if x in self.data and y in self.data[x]:
                    return self.data[x][y]
                else:
                    return 0
            else:
                out = sparse_dict(1,self.shape[1])
                for yi in range(y_start,y_stop,y_step):
                    if x in self.data and yi in self.data[x]:
                        out.set(0,yi,self.data[x][yi])
                return out

        elif not isinstance(y,slice):
            out = sparse_dict(self.shape[0],1)
            for xi in range(x_start, x_stop, x_step):
                if xi in self.data and y in self.data[x]:
                    out.set(xi, 0, self.data[xi][y])
            return out
        else:
            if x.step is None:
                mx = int((x.stop-x.start))
            else:
                mx = int((x.stop-x.start)/x.step)
            if y.step is None:
                my = int((y.stop-y.start))
            else:
                my = int((y.stop-y.start)/y.step)
            out = sparse_dict(mx,my)
            for i, xi in enumerate(range(x_start, x_stop, x_step)):
                for j,yi in enumerate(range(y_start,y_stop,y_step)):
                    out[i,j] = self.data[xi][yi]
            return out

    def logical_and(self,other):
        if self.mx != other.mx or self.my != other.my:
            raise ValueError('Sparse matrices must share sizes')
        common = sparse_dict(self.mx,self.my)
        for x in self.data:
            if x in other.data:
                for y in self.data[x]:
                    if y in other.data[x]:
                        common.set(x,y,1.)
        return common

    def sum(self,axis=-1):
        if axis == 0:
            s = sparse_dict(1,self.my)
            for x in self.data:
                for y in self.data[x]:
                    if s.data and y in s.data[0]:
                        s.data[0][y] += 1
                    else:
                        s.set(0,y,1.)
        else:
            s = sparse_dict(self.mx, 1)
            for x in self.data:
                s.set(x, 0, sum(self.data[x].values()))
            if axis != 1:
                return sum([y for x in s.data.values() for y in x.values()])
        return s


    def any(self,axis):
        if axis == 0:
            a = np.zeros(self.my,dtype=bool)
            for x in self.data:
                for y in self.data[x]:
                    a[y] = 1
        else:
            a = np.zeros(self.mx,dtype=bool)
            for x in self.data:
                a[x] = 1
        return a

    def argmax(self,axis):
        if axis <= 1:
            # exit()
            a = np.zeros(self.my, dtype=np.int16)
            for x in self.data:
                for y in self.data[x]:
                    try:
                        if a[y] not in self.data or (y in self.data[a[y]] and self.data[x][y] > self.data[a[y]][y]):
                            a[y] = x
                    except Exception as E:
                        print(y, x, self.my)
                        print(a[y])
                        print(E)
                        exit()
        else:
            a = np.zeros(self.mx, dtype=np.int16)
            for x in self.data:
                a[x] = 0
                for y in self.data[x]:
                    if a[x] not in self.data[x] or self.data[x][y] > self.data[x][a[x]]:
                        a[x] = y
        return a

    def dict_avg(self,B):
        C = self.copy()

        for x in B.data:
            for y in B.data[x]:
                if x in C.data and y in C.data[x]:
                    C.data[x][y] += B.data[x][y]
                else:
                    C.set(x,y,B.data[x][y])
        for x in C.data:
            for y in C.data[x]:
                C.data[x][y] /= 2
        return C

    def rem(self,x=None,y=None):
        if x is not None and x in self.data:
            if y is not None and y in self.data[x]:
                del self.data[x][y]
            if not self.data[x] or y is None:
                del self.data[x]
        if x is None and y is not None:
            for xi in list(self.data.keys()):
                self.rem(xi,y)

    def todense(self):
        dense = np.zeros(self.shape)
        for x in self.data:
            for y in self.data[x]:
                dense[x,y] = self.data[x][y]
        return dense

    def copy(self):
        cp = sparse_dict(*self.shape)
        cp.data = dict(self.data)
        return cp

    def fromarray(self,a):
        if isinstance(a,list):
            a = np.array(a).reshape((-1,1))
        d = []
        r = []
        c = []
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j]:
                    d.append(a[i,j])
                    r.append(i)
                    c.append(j)
        self.shape = a.shape
        self.mx = self.shape[0]
        self.my = self.shape[1]
        self.fill(d, r, c)
        return


def random_sparse(shape):
    a = np.random.random(shape)
    a = a > 0.6
    sd = sparse_dict(0,0)
    sd.fromarray(a)
    return sd, a

if __name__ == '__main__':
    s = (5,7)
    a,aa = random_sparse(s)
    b,bb = random_sparse(s)


    print('sum 0')
    print(a.sum(0).todense())
    print(aa.sum(0))
    print('sum 1')
    print(a.sum(1).todense())
    print(aa.sum(1))
    print('sum -1')
    print(a.sum())
    print(aa.sum())
    print('any 0')
    print(a.any(0))
    print(aa.any(0))
    print('any 2')
    print(a.any(1))
    print(aa.any(1))
    c = a.logical_and(b)
    cd = c.todense()
    print('LOG AND',np.array_equal(cd,np.logical_and(aa,bb)))
    print('here')
    d = a[b]
    dd = aa[bb]
    print('there')
    print(np.array_equal(a.todense(),aa))
    print(np.array_equal(b.todense(),bb))
    print(dd.shape)
    print(d.sum(),dd.sum())







