import numpy as np
from multiprocessing import Pool
from munkres import Munkres
import matplotlib.pyplot as plt
from sparse_dict import sparse_dict

def validate(matching):
    cols = matching.sum(0)
    rows = matching.sum(1)
    # print(cols.shape)
    n_cols = int(cols.sum())
    n_rows = int(rows.sum())
    n = min(matching.shape)
    return n == n_cols == n_rows


def common_edges(A,B):
    return A.logical_and(B)

def complete_matching(_matching):
    matching = _matching.copy()
    cols = matching.any(0)
    rows = matching.any(1)
    # print(cols.shape)
    n_cols = np.sum(np.logical_not(cols))
    n_rows = np.sum(np.logical_not(rows))
    n = np.minimum(n_rows,n_cols)
    match_cols = np.random.permutation(n_cols)
    match_rows = np.random.permutation(n_rows)
    if n < n_cols:
        match_cols = np.random.choice(match_cols,n)
    if n < n_rows:
        match_rows = np.random.choice(match_rows,n)
    for ci, col in enumerate(cols):
        if col:
            match_cols[match_cols >= ci] += 1

    for ri, row in enumerate(rows):
        if row:
            match_rows[match_rows >= ri] += 1

    for mr, mc in zip(match_rows, match_cols):
        matching.set(mr,mc,1)
    if not validate(matching):
        print('Invalid Matching')
        exit()
    return matching


def evaluate(matching, weights):
    return weights[matching].sum()

def single_point_crossover(A,B):
    A = A.tolist()
    B = B.tolist()
    pt = np.random.randint(0,len(A))
    a1 = A[:pt]
    a2 = A[pt:]
    b1 = B[:pt]
    b2 = B[pt:]
    a_pts = list(set(a1).intersection(set(b2)))
    b_pts = list(set(b1).intersection(set(a2)))
    for a_pt, b_pt in zip(a_pts,b_pts):
        bi = b2.index(a_pt)
        b2[bi] = b_pt
        ai = a2.index(b_pt)
        a2[ai] = a_pt
    A_ = a1 + b2
    B_ = b1 + a2
    return A_, B_

def single_point_breeding(population, weights):
    idx_axis = np.array(list(range(weights.shape[0])))
    idx_chromos = [p.argmax(axis=1) for p in population]

    n = len(population)
    parent_pairs = [(idx_chromos[i], idx_chromos[j]) for i in range(n-1) for j in range(i+1,n)]
    next_gen_idx_chromo = [indv for a,b in parent_pairs for indv in single_point_crossover(a,b)]
    next_gen = []
    for chromo in next_gen_idx_chromo:
        # m = np.zeros_like(weights,dtype=bool)
        m = sparse_dict(*weights.shape)
        for r,c in zip(idx_axis,chromo):
            m.set(r,c,1)
        if not validate(m):
            print('Invalid Matching')
            exit()
        next_gen.append(m)
    next_gen_fitness = [evaluate(m,weights) for m in next_gen]
    return next_gen_fitness, next_gen

def spawn_gen(pop_size,weights):
    population = [complete_matching(sparse_dict(*weights.shape)) for _ in range(pop_size)]
    fitness = np.array([evaluate(p,weights) for p in population])
    return fitness, population

def bit_mutation(matching):
    n = matching.shape[1]
    nb = 1
    m_ = matching.copy()
    for _ in range(nb):
        pts = np.random.choice(np.array(range(n)),2,replace=False)
        c = [m_[i,pts[0]] for i in range(m_.shape[0])]
        for i in m_:
            if pts[0] in m_[i]:
                v = bool(m_[i][pts[0]])
            else:
                v = 0
            if pts[1] in m_[i]:
                m_.set(i,pts[0],m_[i][pts[1]])
                m_.set(i, pts[1], v)
            m_.set(i,pts[1],c[i])
    return m_

def selection(base, children, mutated):
    base_fitness, base_population = base
    n = len(base_population)
    nb = int(0.3*n)
    nc = int(0.6*n)
    nm = n-nc-nb
    base = list(zip(base_fitness, base_population))
    child_fitness, child_population = children
    mutated_fitness, mutated_population = mutated
    children = list(zip(child_fitness, child_population))
    mutated = list(zip(mutated_fitness, mutated_population))

    base.sort(reverse=True,key=lambda x: x[0])

    n_child_fitness = np.array(child_fitness)/np.sum(child_fitness)

    n_mutated_fitness = np.array(mutated_fitness)/np.sum(mutated_fitness)

    next_gen = base[:nb]
    next_gen += [children[c] for c in np.random.choice(list(range(len(n_child_fitness))),nc,replace=False, p = n_child_fitness)]
    next_gen += [mutated[m] for m in np.random.choice(list(range(len(n_mutated_fitness))),nm,replace=False, p = n_mutated_fitness)]

    next_gen_fitness, next_gen_population = list(map(list, zip(*next_gen)))
    return next_gen_fitness, next_gen_population

def YaoEvo(*weights_,generations,population_size):
    nw = len(weights_)
    weights = np.zeros_like(weights_[0])
    for i in range(1,nw): weights += weights_[i]
    weights /= nw
    history = [[] for _ in range(nw)]
    pop_history = []
    # step 1
    fitness_ = [[] for _ in range(nw)]
    fitness, population = spawn_gen(population_size, weights)
    for i in range(nw):
        fitness_[i] = [evaluate(p, weights_[i]) for p in population]
    for gen in range(generations):
        pop_fit = list(zip(*fitness_, population))
        top_indvs = []
        pop_fit.sort(reverse=True, key=lambda x: np.mean(x[:nw]))
        top_indvs.append(pop_fit[0])
        pop_history.append(fitness_)
        for i in range(nw):
            pop_fit.sort(reverse=True, key=lambda x: x[i])
            top_indvs.append(pop_fit[0])

        for i in range(nw):
            history[i].append(top_indvs[1+i][:nw])

        child_fitness, child_population = single_point_breeding(population, weights)
        mutated_population = [bit_mutation(m) for m in population]
        mutated_fitness = [evaluate(m,weights) for m in mutated_population]

        fitness, population = selection((fitness,population),(child_fitness, child_population), (mutated_fitness, mutated_population))
        for i in range(nw):
            fitness_[i] = [evaluate(p, weights_[i]) for p in population]

    pop_fit = list(zip(*fitness_, population))
    top_indvs = []
    for i in range(nw):
        pop_fit.sort(reverse=True, key=lambda x: x[i])
        top_indvs.append(pop_fit[0])
    pop_history.append(fitness_)
    return top_indvs[1], history, pop_history

def find_and_eval_matching(weights1):
    m = Munkres()
    matching = m.compute(np.copy(weights1))
    matching_weight1 = []
    match_mat = np.zeros_like(weights1, dtype=bool)
    for j, r in matching:
        matching_weight1.append(weights1[j, r])
        match_mat[j, r] = 1
    return matching, matching_weight1, match_mat

if __name__ == '__main__':
    weights1 = np.random.random((25,25))
    weights2 = np.random.random((25,25))
    nweights = np.max(weights1)-weights1
    _,_,oM = find_and_eval_matching(nweights)
    optimal1 = evaluate(oM,weights1)
    optimal2 = evaluate(oM,weights2)
    print('Optimal:',optimal1)
    match, history, pop_hist = YaoEvo(weights1, weights2,generations=300,population_size=30)
    h1,h2 = history

    print('Optimal: {}, {}'.format(optimal1, optimal2))
    new_optimal1 = evaluate(match[-1], weights1)
    new_optimal2 = evaluate(match[-1], weights2)
    prcnt_f1 = (new_optimal1 - optimal1) / optimal1
    prcnt_f2 = (new_optimal2 - optimal2) / optimal2
    print('New Optimal: {}, {}'.format(new_optimal1, new_optimal2))
    print('dF1: {}%, dF2: {}%'.format(prcnt_f1, prcnt_f2))


    h11, h12 = list(map(list, zip(*h1)))
    h21, h22 = list(map(list, zip(*h2)))
    x = list(range(len(h1)))
    plt.plot(x, h11, 'g--', label='Highest F1, F1')
    plt.plot(x, h12, 'g-.', label='Highest F1, F2')
    plt.plot(x, h21, 'm--', label='Highest F2, F1')
    plt.plot(x, h22, 'm-.', label='Highest F2, F2')
    plt.plot([0, len(h1)], [optimal1, optimal1], 'r--', label='Optimal F1')
    plt.plot([0, len(h1)], [optimal2, optimal2], 'b--', label='Sub-Optimal F2')
    plt.legend()
    plt.title('Scalarized Genetic Algorithm, Yao et. al.')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.show()
    plt.show()







