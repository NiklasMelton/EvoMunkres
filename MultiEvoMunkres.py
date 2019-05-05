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

def cross_breed_population(population,fitness1, fitness2,num_to_gen,m_rate=0.01):
    sf1 = np.sum(fitness1)
    sf2 = np.sum(fitness2)
    nfit1 = fitness1/sf1
    nfit2 = fitness2/sf2
    nfit3 = np.add(fitness1,fitness2)/2
    nfit3 /= np.sum(nfit3)
    idx = list(range(len(population)))
    num_to_gen1 = int(num_to_gen/4)
    num_to_gen2 = num_to_gen-(3*num_to_gen1)
    parent_idx = [(np.random.choice(idx,p=nfit1),np.random.choice(idx,p=nfit1)) for _ in range(num_to_gen1)]
    parent_idx += [(np.random.choice(idx,p=nfit2),np.random.choice(idx,p=nfit2)) for _ in range(num_to_gen1)]
    parent_idx += [(np.random.choice(idx,p=nfit1),np.random.choice(idx,p=nfit2)) for _ in range(num_to_gen1)]
    parent_idx += [(np.random.choice(idx,p=nfit3),np.random.choice(idx,p=nfit3)) for _ in range(num_to_gen2)]
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

def breed_and_evaluate(population, fitness1, fitness2,weights1, weights2, num_to_gen, m_rate=0.01):
    next_gen = []
    children = cross_breed_population(population,fitness1,fitness2,num_to_gen,m_rate)
    # p = Pool(4)
    # children_evaluations = list(p.map(pevaluate,[(c,weights) for c in children]))
    # p.close()
    child_evaluations1 = [evaluate(c,weights1) for c in children]
    child_evaluations2 = [evaluate(c,weights2) for c in children]
    child_fit = list(zip(child_evaluations1, child_evaluations2,children))

    next_gen += child_fit
    return next_gen

def cull(population, fitness1, fitness2, pop_limit):
    npop_limit = min(len(population),pop_limit)
    idx = list(range(len(population)))
    pop_fit1 = list(zip(fitness1, fitness2, population, idx))
    pop_fit2 = list(zip(fitness1, fitness2, population, idx))
    pop_fit3 = list(zip(fitness1, fitness2, population, idx, np.add(fitness1,fitness2)/2))

    pop_fit1.sort(reverse=True,key=lambda x: x[0])
    pop_fit2.sort(reverse=True,key=lambda x: x[1])
    pop_fit3.sort(reverse=True,key=lambda x: x[-1])
    npop_limit1 = int(npop_limit/2)
    npop_fit = pop_fit3[:npop_limit1]
    npop_fit_idx = [x[-1] for x in npop_fit]
    npi = 0
    while len(npop_fit) < npop_limit:
        if pop_fit3[npi][-2] not in npop_fit_idx:
            npop_fit.append(pop_fit3[npi][:-1])
            npop_fit_idx.append(pop_fit3[npi][-2])
        npi += 1

    nfitness1, nfitness2, npopulation, _ = list(map(list, zip(*npop_fit)))
    return nfitness1, nfitness2, npopulation


def spawn_gen(pop_size,weights1, weights2):
    population = [complete_matching(np.zeros_like(weights1,dtype=bool)) for _ in range(pop_size)]
    fitness1 = np.array([evaluate(p,weights1) for p in population])
    fitness2 = np.array([evaluate(p,weights2) for p in population])
    return fitness1, fitness2, population

def breed_and_cull(population, fitness1, fitness2, weights1,weights2,population_size, keep_top_num,birthrate, m_rate):
    idx = list(range(len(population)))
    pop_fit1 = list(zip(fitness1, fitness2, population, idx))
    pop_fit2 = list(zip(fitness1, fitness2, population, idx))
    pop_fit1.sort(reverse=True, key=lambda x: x[0])
    pop_fit2.sort(reverse=True, key=lambda x: x[1])

    keep_top_num1 = int(keep_top_num/2)

    next_gen = pop_fit1[:keep_top_num1]
    next_gen_idx = [x[-1] for x in next_gen]
    ngi = 0
    while len(next_gen) < keep_top_num:
        if pop_fit2[ngi][-1] not in next_gen_idx:
            next_gen.append(pop_fit2[ngi])
            next_gen_idx.append(pop_fit2[ngi][-1])
        ngi += 1
    next_gen = [x[:-1] for x in next_gen]
    # mutated_next_gen = [mutate(ng,m_rate) for _,_,ng in next_gen for _ in range(3)]
    # next_gen += [(evaluate(ng,weights1),evaluate(ng,weights2),ng) for ng in mutated_next_gen]
    next_gen += breed_and_evaluate(population, fitness1, fitness2, weights1,weights2, int(population_size * birthrate), m_rate)
    next_gen_fitness1, next_gen_fitness2, next_gen_population = list(map(list, zip(*next_gen)))
    fitness1, fitness2, population = cull(next_gen_population, next_gen_fitness1, next_gen_fitness2, population_size)
    return fitness1, fitness2, population

def MultiEvoMunkres(weights1,weights2,generations,population_size,birthrate=2.0,keep_top_num=1,m_rate=0.01):
    history1 = []
    history2 = []
    history3 = []
    fitness1,fitness2, population = spawn_gen(population_size,weights1,weights2)
    no_change_count1= 0
    no_change_count2= 0
    dm_rate = m_rate
    last_fitness1 = 0
    last_fitness2 = 0
    for gen in range(generations):
        pop_fit1 = list(zip(fitness1, fitness2, population))
        pop_fit2 = list(zip(fitness1, fitness2, population))
        pop_fit3 = list(zip(fitness1, fitness2, population, np.add(fitness1,fitness2)/2))
        pop_fit1.sort(reverse=True,key=lambda x: x[0])
        pop_fit2.sort(reverse=True,key=lambda x: x[1])
        pop_fit3.sort(reverse=True,key=lambda x: x[-1])
        top_indv1 = pop_fit1[0]
        top_indv2 = pop_fit2[0]
        top_indv3 = pop_fit3[0]
        print('Gen {}, Top1 F1: {}, F2: {} ---- Top2 F1: {}, F2: {} ---- Top3 F1: {}, F2: {}'.format(gen,top_indv1[0],top_indv1[1],top_indv2[0],top_indv2[1],top_indv3[0],top_indv3[1]))
        history1.append(top_indv1[0:2])
        history2.append(top_indv2[0:2])
        history3.append(top_indv3[0:2])

        if top_indv1[0] == last_fitness1:
            no_change_count1 += 1
        else:
            no_change_count1 = 0
            dm_rate = m_rate
        if top_indv2[0] == last_fitness2:
            no_change_count2 += 1
        else:
            no_change_count2 = 0
            dm_rate = m_rate
        if no_change_count1 > 2 or no_change_count2 > 2:
            dm_rate = min(0.3,dm_rate+0.01)
            no_change_count1 = 0
            no_change_count2 = 0
        last_fitness1 = top_indv1[0]
        last_fitness2 = top_indv2[0]
        print('Mutation Rate: {}'.format(dm_rate))
        fitness1, fitness2 , population = breed_and_cull(population, fitness1, fitness2, weights1,weights2, population_size, keep_top_num, birthrate, m_rate)
    pop_fit1 = list(zip(fitness1, fitness2, population))
    pop_fit1.sort(reverse=True, key=lambda x: x[0])
    top_indv = pop_fit1[0]
    return top_indv, history1, history2, history3

def find_and_eval_matching(weights1,weights2=None):
    m = Munkres()
    matching = m.compute(np.copy(weights1))
    matching_weight1 = []
    matching_weight2 = []
    match_mat = np.zeros_like(weights1, dtype=bool)
    for j, r in matching:
        matching_weight1.append(weights1[j, r])
        if weights2 is not None:
            matching_weight2.append(weights2[j, r])
        match_mat[j, r] = 1
    if weights2 is None:
        return matching, matching_weight1, match_mat
    else:
        return matching, matching_weight1, matching_weight2, match_mat

if __name__ == '__main__':
    weights1 = np.random.random((25,25))
    weights2 = np.random.random((25,25))
    nweights1 = np.max(weights1)-weights1
    _,_,oM = find_and_eval_matching(nweights1)
    optimal1 = evaluate(oM,weights1)
    optimal2 = evaluate(oM,weights2)
    print('Optimal: {}, {}'.format(optimal1, optimal2))
    match, h1, h2, h3 = MultiEvoMunkres(weights1,weights2,300,20,birthrate=40.0,keep_top_num=3,m_rate=0.05)

    h11 = [a for a,b in h1]
    h12 = [b for a,b in h1]
    h21 = [a for a,b in h2]
    h22 = [b for a,b in h2]
    h31 = [a for a, b in h3]
    h32 = [b for a, b in h3]
    nh = len(h11)
    x = list(range(nh))
    plt.plot(x,h11,'g-',x,h12,'b-',x,h21,'g-.',x,h22,'b-.',[0,nh],[optimal1,optimal1],'g--',[0,nh],[optimal2,optimal2],'b--',x,h31,'r-',x,h32,'k-')
    plt.show()







