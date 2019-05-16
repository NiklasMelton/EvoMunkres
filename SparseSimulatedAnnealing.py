import numpy as np
import matplotlib.pyplot as plt
from munkres import Munkres
from sparse_dict import sparse_dict

def evaluate(matching, weights):
    return weights[matching].sum()

def spawn_gen(pop_size,weights):
    population = [complete_matching(sparse_dict(*weights.shape)) for _ in range(pop_size)]
    fitness = [evaluate(p,weights) for p in population]
    return fitness, population

def validate(matching):
    cols = matching.sum(0)
    rows = matching.sum(1)
    # print(cols.shape)
    n_cols = int(cols.sum())
    n_rows = int(rows.sum())
    n = min(matching.shape)

    return n == n_cols == n_rows


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
        matching[mr,mc] = 1
    if not validate(matching):
        print('Invalid Matching')
        raise ValueError('Invalid Matching')
    return matching


def mutate(_matching, p=0.01):
    matching = _matching.copy()
    # print('s',matching.shape)
    n = np.max(matching.shape)
    flips = np.random.random(n) < p
    if matching.shape[0] >= matching.shape[1]:
        for f in flips:
            matching.rem(x=f,y=None)
    else:
        for f in flips:
            matching.rem(x=None,y=f)
    return complete_matching(matching)

def Anneal(weights,population_size,t0=1000, tr=0.003,m_rate=0.3):
    history = []
    t = t0
    # step 1
    fitness, population = spawn_gen(population_size,weights)
    while t > 1e-5:
        pop_fit = list(zip(fitness, population))
        pop_fit.sort(reverse=True,key=lambda x: x[0])
        top_indv = pop_fit[0]
        print('Temp: {}, Top Fitness: {}'.format(t,top_indv[0]))
        history.append(top_indv[0])
        mutated_population = [mutate(m,m_rate) for m in population]
        mutated_fitness = [evaluate(m,weights) for m in mutated_population]
        next_gen = [[mf,mp] if mf > f or np.exp((mf-f)/t) > np.random.random() else [f,p] for f,p,mf,mp in zip(fitness,population,mutated_fitness,mutated_population)]
        t *= 1-tr
        fitness, population = list(map(list, zip(*next_gen)))

    pop_fit = list(zip(fitness, population))
    pop_fit.sort(reverse=True, key=lambda x: x[0])
    top_indv = pop_fit[0]
    return top_indv, history

def MultiAnneal(*weights_,population_size=100,t0=1000, tr=0.003,m_rate=0.3):

    weights = [w.copy() for w in weights_]
    nw = len(weights)
    history = [[] for _ in weights]
    pop_history = []
    t = t0
    # step 1
    fitness = [[] for _ in weights]
    fitness[0], population = spawn_gen(population_size,weights[0])
    for i in range(1,nw):
        fitness[i] = [evaluate(p,weights[i]) for p in population]
    while t > 1e-5:
        pop_history.append(list(fitness))
        # print('FIT 1',len(fitness),len(pop_history[0]))
        pop_fit = list(zip(*fitness, population))
        top_indvs = []
        for i in range(nw):
            pop_fit.sort(reverse=True, key=lambda x: x[i])
            top_indvs.append(pop_fit[0])

        print('Temp: {}, Top Fitness: {}'.format(t,[t[:nw] for t in top_indvs]))
        for i in range(nw):
            history[i].append(top_indvs[i][:nw])
        mutated_population = [mutate(m,m_rate) for m in population]
        mutated_fitness = [[evaluate(m,weights[i]) for m in mutated_population] for i in range(nw)]
        next_gen = [[mf ,mp] if np.all([mfi > fi for mfi, fi in zip(mf,f)]) or np.all([np.exp((mfi-fi)/t) > np.random.random() for mfi, fi in zip(mf,f)])
                    else [f,p] for f,p,mf,mp in zip(fitness,population,mutated_fitness,mutated_population)]
        t *= 1-tr
        fitness, population = list(map(list,zip(*next_gen)))
        # print('FIT 2', len(fitness), len(pop_history[0]))

    pop_fit = list(zip(fitness[0], population))
    pop_fit.sort(reverse=True, key=lambda x: x[0])
    top_indv = pop_fit[0]
    pop_history.append(fitness)
    return top_indv, history, pop_history

def ScalarMultiAnneal(*weights_,population_size=100,t0=1000, tr=0.003,m_rate=0.3):

    weights = [w.copy() for w in weights_]
    nw = len(weights)
    history = [[] for _ in weights]
    pop_history = []
    t = t0
    # step 1
    fitness = [[] for _ in weights]
    fitness[0], population = spawn_gen(population_size,weights[0])
    for i in range(1,nw):
        fitness[i] = [evaluate(p,weights[i]) for p in population]
    while t > 1e-5:
        pop_history.append(list(fitness))
        # print('FIT 1',len(fitness),len(pop_history[0]))
        pop_fit = list(zip(*fitness, population))
        top_indvs = []
        for i in range(nw):
            pop_fit.sort(reverse=True, key=lambda x: x[i])
            top_indvs.append(pop_fit[0])

        print('Temp: {}, Top Fitness: {}'.format(t,[t[:nw] for t in top_indvs]))
        for i in range(nw):
            history[i].append(top_indvs[i][:nw])
        mutated_population = [mutate(m,m_rate) for m in population]
        mutated_fitness = [[evaluate(m,weights[i]) for m in mutated_population] for i in range(nw)]
        # next_gen = [[mf ,mp] if np.all([mfi > fi for mfi, fi in zip(mf,f)]) or np.all([np.exp((mfi-fi)/t) > np.random.random() for mfi, fi in zip(mf,f)])
        #             else [f,p] for f,p,mf,mp in zip(fitness,population,mutated_fitness,mutated_population)]
        next_gen = [[mf, mp] if np.mean(mf) > np.mean(f) or np.exp((np.mean(mf) - np.mean(f)) / t) > np.random.random()
                    else [f,p] for f,p,mf,mp in zip(fitness,population,mutated_fitness,mutated_population)]
        t *= 1-tr
        fitness, population = list(map(list,zip(*next_gen)))
        # print('FIT 2', len(fitness), len(pop_history[0]))

    pop_fit = list(zip(fitness[0], population))
    pop_fit.sort(reverse=True, key=lambda x: x[0])
    top_indv = pop_fit[0]
    pop_history.append(fitness)
    return top_indv, history, pop_history


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
    import pickle
    # n = 1000
    # data = pickle.load(open('matching.pckl', 'rb'))
    # print(data.keys())
    # oM = data['matching']
    # weights1 = data['weights1']
    # weights2 = data['weights2']
    #
    # sweights1 = sparse_dict(0, 0)
    # sweights2 = sparse_dict(0, 0)
    # OM = sparse_dict(0, 0)
    # sweights1.fromarray(weights1)
    # sweights2.fromarray(weights2)
    # OM.fromarray(oM)

    data = pickle.load(open('skill_fitness.pckl','rb'))
    mx = max(data[1])+1
    my = max(data[2])+1
    print('MX',mx)
    print('MY',my)
    sweights1 = sparse_dict(mx,my)
    sweights2 = sparse_dict(mx,my)
    for d,r,c in zip(*data):
        sweights1.set(r,c,d)
        sweights2.set(r,c,np.random.random())
    pickle.dump({'fitness':sweights1,'value':sweights2},open('sparse_fitness.pckl','wb'))
    # data = pickle.load(open('sparse_fitness.pckl','rb'))
    # sweights1 = data['fitness']
    # sweights2 = data['value']
    # soptimal1 = evaluate(oM, weights1)
    # optimal1 = evaluate(OM, sweights1)
    # optimal2 = evaluate(OM, sweights2)
    # print('Optimal: {}, {}'.format(optimal1, optimal2))
    # match, history = Anneal(weights,100,t0=5, tr=0.003,m_rate=0.05)
    # match, history, pop_history = ScalarMultiAnneal(weights1,weights2,population_size=30,t0=10000, tr=0.003,m_rate=0.3)
    match, history, pop_history = MultiAnneal(sweights1,sweights2,population_size=100,t0=1, tr=0.003,m_rate=0.001)
    h1, h2 =  history
    # print('Optimal:', optimal1, optimal2)
    h11, h12 = list(map(list, zip(*h1)))
    h21, h22 = list(map(list, zip(*h2)))
    x = list(range(len(h1)))
    plt.plot(x,h11,'g--',label ='Highest F1, F1')
    plt.plot(x,h12,'g-.',label ='Highest F1, F2')
    plt.plot(x,h21,'m--',label ='Highest F2, F1')
    plt.plot(x,h22,'m-.',label ='Highest F2, F2')
    # plt.plot([0,len(h1)],[optimal1,optimal1],'r--',label='Optimal F1')
    # plt.plot([0,len(h1)],[optimal2,optimal2],'b--',label='Sub-Optimal F2')
    plt.legend()
    plt.title('Scalarized Simulated Annealing')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')

    # print('Optimal: {}, {}'.format(optimal1, optimal2))
    new_optimal1 = evaluate(match[-1], sweights1)
    new_optimal2 = evaluate(match[-1], sweights2)
    # prcnt_f1 = (new_optimal1 - optimal1) / optimal1
    # prcnt_f2 = (new_optimal2 - optimal2) / optimal2
    print('New Optimal: {}, {}'.format(new_optimal1, new_optimal2))
    # print('dF1: {}%, dF2: {}%'.format(prcnt_f1, prcnt_f2))

    plt.show()
