import numpy as np
from multiprocessing import Pool
from munkres import Munkres
import matplotlib.pyplot as plt


def validate(matching):
    cols = np.sum(matching, 0)
    rows = np.sum(matching, 1)
    # print(cols.shape)
    n_cols = np.sum(cols)
    n_rows = np.sum(rows)
    return np.array_equal((n_rows,n_cols),matching.shape)

def common_edges(A,B):
    return np.logical_and(A,B)

def complete_matching(_matching):
    matching = np.copy(_matching)
    cols = np.any(matching, 0)
    rows = np.any(matching, 1)
    # print(cols.shape)
    n_cols = np.sum(np.logical_not(cols))
    n_rows = np.sum(np.logical_not(rows))
    n = np.minimum(n_rows,n_cols)
    match_cols = np.random.permutation(n_cols)
    match_rows = np.random.permutation(n_rows)
    if n < n_cols:
        match_cols = np.choose(match_cols,n)
    if n < n_rows:
        match_rows = np.choose(match_rows,n)
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
        exit()
    return matching


def mutate(_matching, p=0.01):
    matching = np.copy(_matching)
    # print('s',matching.shape)
    n = np.max(matching.shape)
    flips = np.random.random(n) < p
    if matching.shape[0] >= matching.shape[1]:
        matching[flips,:] = 0
    else:
        matching[:,flips] = 0
    return complete_matching(matching)

def mate(A,B,m_rate=0.01):
    return mutate(common_edges(A,B),p=m_rate)

def breed(A,B,m_rate=0.01,n_offspring=1):
    if n_offspring == 1:
        return mate(A,B,m_rate)
    else:
        return [mate(A,B,m_rate) for _ in range(n_offspring)]

def pbreed(args):
    return breed(*args)

def pmate(args):
    return mate(*args)

def breed_population(population,fitness,num_to_gen,m_rate=0.01):
    fitness = np.array(fitness)
    nfit = fitness/np.sum(fitness)
    idx = list(range(len(population)))
    parent_idx = [(np.random.choice(idx,p=nfit),np.random.choice(idx,p=nfit)) for _ in range(num_to_gen)]
    parent_pairs = [(population[i],population[j],m_rate) for i,j in parent_idx]
    # p = Pool(1)
    # next_gen = list(p.map(pmate,parent_pairs))
    # p.close()
    next_gen = [pmate(a) for a in parent_pairs]
    return next_gen

def evaluate(matching, weights):
    return np.sum(weights[matching])

def pevaluate(args):
    return evaluate(*args)

def breed_and_evaluate(population, fitness,weights, num_to_gen, m_rate=0.01):
    fitness = np.array(fitness)
    next_gen = []
    children = breed_population(population,fitness,num_to_gen,m_rate)
    # p = Pool(4)
    # children_evaluations = list(p.map(pevaluate,[(c,weights) for c in children]))
    # p.close()
    children_evaluations = [evaluate(c,weights) for c in children]
    child_fit = list(zip(children_evaluations,children))

    next_gen += child_fit
    return next_gen

def cull(population, fitness, pop_limit, lowest=False):
    npop_limit = min(len(population),pop_limit)
    pop_fit = list(zip(fitness, population))
    if lowest:
        pop_fit.sort(reverse=True,key=lambda x: x[0])
        npop_fit = pop_fit[:npop_limit]
    else:
        nfit = fitness/np.sum(fitness)
        idx = list(range(len(population)))
        pop_fit_idx = np.random.choice(idx,npop_limit,p=nfit,replace=False)
        npop_fit = [pop_fit[i] for i in pop_fit_idx]
    nfitness, npopulation = list(map(list, zip(*npop_fit)))
    del population
    del fitness
    return nfitness, npopulation


def spawn_gen(pop_size,weights):
    population = [complete_matching(np.zeros_like(weights,dtype=bool)) for _ in range(pop_size)]
    fitness = np.array([evaluate(p,weights) for p in population])
    return fitness, population

def breed_and_cull(population, fitness, weights, population_size, keep_top_num,birthrate, m_rate, lowest=False):
    pop_fit = list(zip(fitness, population))
    next_gen = pop_fit[:keep_top_num]
    next_gen += breed_and_evaluate(population, fitness, weights, int(population_size * birthrate), m_rate)
    next_gen_fitness, next_gen_population = list(map(list, zip(*next_gen)))
    fitness, population = cull(next_gen_population, next_gen_fitness, population_size, lowest=lowest)
    return fitness, population

def EvoMunkres(weights,generations,population_size,birthrate=2.0,keep_top_num=1,m_rate=0.01):
    history = []
    fitness, population = spawn_gen(population_size,weights)
    no_change_count= 0
    dm_rate = m_rate
    last_fitness = 0
    for gen in range(generations):
        pop_fit = list(zip(fitness, population))
        pop_fit.sort(reverse=True,key=lambda x: x[0])
        top_indv = pop_fit[0]
        print('Gen {}, Top Fitness: {}'.format(gen,top_indv[0]))
        history.append(top_indv[0])

        if top_indv[0] == last_fitness:
            no_change_count += 1
        else:
            no_change_count = 0
            dm_rate = m_rate
        if no_change_count > 2:
            dm_rate = min(0.3,dm_rate+0.025)
            no_change_count = 0
        last_fitness = top_indv[0]
        print('Mutation Rate: {}'.format(dm_rate))
        fitness, population = breed_and_cull(population, fitness, weights, population_size, keep_top_num, birthrate, m_rate, lowest=True)
    pop_fit = list(zip(fitness, population))
    pop_fit.sort(reverse=True, key=lambda x: x[0])
    top_indv = pop_fit[0]
    return top_indv, history

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
    weights = np.random.random((25,25))
    nweights = np.max(weights)-weights
    _,_,oM = find_and_eval_matching(nweights)
    optimal = evaluate(oM,weights)
    print('Optimal:',optimal)
    match, history = EvoMunkres(weights,100,30,birthrate=30.0,keep_top_num=30,m_rate=0.05)

    plt.plot(list(range(len(history))),history,'g',[0,len(history)],[optimal,optimal],'r')
    plt.show()







