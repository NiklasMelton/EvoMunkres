import numpy as np
import matplotlib.pyplot as plt
from munkres import Munkres

def evaluate(matching, weights):
    return np.sum(weights[matching])


def validate(matching):
    cols = np.sum(matching, 0)
    rows = np.sum(matching, 1)
    # print(cols.shape)
    n_cols = np.sum(cols)
    n_rows = np.sum(rows)
    return np.array_equal((n_rows,n_cols),matching.shape)


def ModMunkres(*weights_):
    weights = [np.copy(w) for w in weights_]
    history = [[] for _ in weights]
    matchings = []
    m = find_matching(weights[0])

    v = np.inf
    e = 0
    while v > 0:
        matchings.append(m)
        rm_pts = []
        v = np.inf
        for i in range(len(history)):
            history[i].append(evaluate(m,weights[i]))
            if i > 0:
                weights[i][weights[0] == 0] = np.inf
                x,y = np.unravel_index(weights[i].argmin(), weights[i].shape)
                weights[i][weights[i] == np.inf] = 0
                if weights[0][x,y] < v:
                    v = weights[0][x,y]
                    rm_pts = (x,y)
        if v > 0:
            weights[0][rm_pts[0],rm_pts[1]] = 0
        m = find_matching(weights[0])
        print('Iter: {}, Fitness: {}'.format(e,[h[-1] for h in history]))
        e += 1
    matchings.append(m)
    for i in range(len(history)):
        history[i].append(evaluate(m, weights[i]))

    return matchings, history, list(map(list, zip(*history)))


def find_matching(weights1):
    nweights1 = np.max(weights1) - weights1
    m = Munkres()
    matching = m.compute(np.copy(nweights1))
    match_mat = np.zeros_like(weights1, dtype=bool)
    for j, r in matching:
        match_mat[j, r] = 1
    return match_mat


if __name__ == '__main__':
    weights1 = np.random.random((25,25))
    weights2 = np.random.random((25,25))

    m = find_matching(weights1)
    optimal1 = evaluate(m,weights1)
    optimal2 = evaluate(m,weights2)

    print('Optimal:',optimal1,optimal2,)
    # match, history = Anneal(weights,100,t0=5, tr=0.003,m_rate=0.05)
    _, [h1, h2], pop_hist = ModMunkres(weights1,weights2)

    print('Optimal:',optimal1,optimal2,)
    x = list(range(len(h1)))
    plt.plot(x,h1,'g',x,h2,'c',[0,len(h1)],[optimal1,optimal1],'r--',[0,len(h1)],[optimal2,optimal2],'b--')
    plt.legend(['Fitness 1', 'Fitness 2', 'Optimal 1', 'Sub-Optimal 2'])
    plt.xlabel('iterations')
    plt.ylabel('Fitness')
    plt.title('Modified Kuhns-Munkres Performance')

    plt.figure()
    plt.plot(h1,h2,'r.')
    _, [h2, h1], pop_hist = ModMunkres(weights2, weights1)
    plt.plot(h1,h2,'b.')
    plt.legend(['Modified A','Modified B'])
    plt.title('Modified Kuhns-Munkres Fitness Space')
    plt.xlabel('Fitness 1')
    plt.ylabel('Fitness 2')

    plt.show()
