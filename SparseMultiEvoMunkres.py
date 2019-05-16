import numpy as np
from multiprocessing import Pool
from munkres import Munkres
import matplotlib.pyplot as plt
import itertools
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
    nfit1 = fitness1/np.sum(fitness1)
    nfit2 = fitness2/np.sum(fitness2)
    nfit3 = np.add(fitness1,fitness2)/2
    nfit3 /= np.sum(nfit3)
    idx = list(range(len(population)))
    num_to_gen1 = int(num_to_gen/2)
    num_to_gen2 = num_to_gen-num_to_gen1
    parent_idx = [(np.random.choice(idx,p=nfit1),np.random.choice(idx,p=nfit2)) for _ in range(num_to_gen1)]
    parent_idx += [(np.random.choice(idx,p=nfit3),np.random.choice(idx,p=nfit3)) for _ in range(num_to_gen2)]
    parent_pairs = [(population[i],population[j],m_rate) for i,j in parent_idx]
    # p = Pool(1)
    # next_gen = list(p.map(pmate,parent_pairs))
    # p.close()
    next_gen = [pmate(a) for a in parent_pairs]
    return next_gen

def evaluate(matching, weights):
    return weights[matching].sum()

def pevaluate(args):
    return evaluate(*args)

def breed_and_evaluate(population, fitness, weights, num_to_gen, m_rate=0.01):
    nw = len(weights)
    all_combos = [[0,1]]
    # all_combos = list(itertools.combinations(list(range(nw)), 2))
    num_to_gen_per = int(num_to_gen / (len(all_combos) + nw))
    children = []
    for i,j in all_combos:
        children += cross_breed_population(population,fitness[i],fitness[j],num_to_gen_per,m_rate)
    for i in range(nw):
        children += cross_breed_population(population, fitness[i], fitness[i], num_to_gen_per, m_rate)
    child_fitness = []
    for i in range(nw):
        child_fitness.append([evaluate(c,weights[i]) for c in children])
    child_fit = list(zip(*child_fitness,children))
    return child_fit

def cull(population, fitness, pop_limit):
    nw = len(fitness)
    npop_limit = min(len(population), pop_limit)
    min_fit = [min(x) for x in zip(*fitness)]
    idx = list(range(len(population)))
    pop_fit = list(zip(min_fit,*fitness,population,idx))
    npop_fit = []
    npop_fit_idx = []
    for i in range(nw):
        pop_fit.sort(reverse=True,key=lambda x: x[i+1])
        if pop_fit[0][-1] not in npop_fit_idx:
            npop_fit_idx.append(pop_fit[0][-1])
            npop_fit.append(pop_fit[0][:-1])
    i = 0
    pop_fit.sort(reverse=True, key=lambda x: x[0])
    while len(npop_fit) < npop_limit:
        if pop_fit[i][-1] not in npop_fit_idx:
            npop_fit.append(pop_fit[i][:-1])
            npop_fit_idx.append(pop_fit[i][-1])
        i += 1
    _,*nfitness, npopulation = list(map(list, zip(*npop_fit)))
    return nfitness, npopulation

# def cull(population, fitness, pop_limit):
#     nw = len(fitness)
#     npop_limit = min(len(population),pop_limit)
#     idx = list(range(len(population)))
#     fitness.append([np.mean(x) for x in zip(*fitness)])
#     pop_fit = list(zip(*fitness, population, idx))
#     pop_fits = [None for _ in range(nw+1)]
#     for i in range(nw+1):
#         pop_fit.sort(reverse=True,key=lambda x: x[i])
#         pop_fits[i] = list(pop_fit)
#     npop_fit = []
#     npop_fit_idx = []
#     i = 0
#     j = -1
#     while len(npop_fit) < npop_limit:
#         if pop_fits[j][i][-1] not in npop_fit_idx:
#             npop_fit.append(pop_fits[j][i][:-1])
#             npop_fit_idx.append(pop_fits[j][i][-1])
#         j = -1
#         i += 1
#     *nfitness, npopulation = list(map(list, zip(*npop_fit)))
#     return nfitness[:-1], npopulation


def spawn_gen(pop_size,weights1,indv0=None):
    if indv0 is None:
        population = [complete_matching(sparse_dict(*weights1.shape)) for _ in range(pop_size)]
    else:
        population = [mutate(indv0.copy(),p=0.01) if np.random.rand() > 0.66 else complete_matching(sparse_dict(*weights1.shape)) for _ in range(pop_size)]
    fitness1 = np.array([evaluate(p,weights1) for p in population])
    return fitness1, population

def breed_and_cull(population, fitness, weights,population_size, keep_top_num,birthrate, m_rate):
    nw = len(weights)
    idx = list(range(len(population)))
    pop_fit = list(zip(*fitness, population, idx))
    pop_fit.sort(reverse=True, key=lambda x: x[0])

    keep_top_num_per = int(keep_top_num/nw)

    next_gen = pop_fit[:keep_top_num_per]
    next_gen_idx = [x[-1] for x in next_gen]
    for i in range(1,nw):
        pop_fit.sort(reverse=True, key=lambda x: x[i])
        ngi = 0
        while len(next_gen) < min((i+1)*keep_top_num_per,keep_top_num):
            if pop_fit[ngi][-1] not in next_gen_idx:
                next_gen.append(pop_fit[ngi])
                next_gen_idx.append(pop_fit[ngi][-1])
            ngi += 1
    next_gen = [x[:-1] for x in next_gen]
    # mutated_next_gen = [mutate(ng,m_rate) for _,_,ng in next_gen for _ in range(3)]
    # next_gen += [(evaluate(ng,weights1),evaluate(ng,weights2),ng) for ng in mutated_next_gen]
    next_gen += breed_and_evaluate(population, fitness, weights, int(population_size * birthrate), m_rate)
    *next_gen_fitness, next_gen_population = list(map(list, zip(*next_gen)))
    fitness, population = cull(next_gen_population, next_gen_fitness, population_size)
    return fitness, population

def random_breed(population, fitness, weights,population_size, keep_top_num,birthrate, m_rate):
    idx = list(range(len(population)))
    pop_fit = list(zip(*fitness, population, idx))
    nw = len(weights)
    next_gen_fitness, next_gen_population = cull(population,fitness,keep_top_num)
    next_gen = list(zip(*next_gen_fitness, next_gen_population))
    gf = [[] for _ in range(nw)]
    gf[0],gp = spawn_gen(int(population_size * birthrate), weights[0])
    for i in range(1,nw):
        gf[i] = [evaluate(p, weights[i]) for p in gp]
    next_gen += list(zip(*gf,gp))
    *next_gen_fitness, next_gen_population = list(map(list, zip(*next_gen)))
    fitness, population = cull(next_gen_population, next_gen_fitness, population_size)
    return fitness, population


def MultiEvoMunkres(*weights,generations,population_size,birthrate=2.0,keep_top_num=1,m_rate=0.01,indv0=None):
    nw = len(weights)
    history = [[] for _ in range(nw)]
    pop_history = []
    fitness = [[] for _ in range(nw)]
    fitness[0], population = spawn_gen(population_size, weights[0],indv0=indv0)
    for i in range(1,nw):
        fitness[i] = [evaluate(p, weights[i]) for p in population]
    no_change_count = [0 for _ in range(nw)]
    dm_rate = m_rate
    last_fitness = [0 for _ in range(nw)]
    for gen in range(generations):
        pop_history.append(fitness)
        top_indvs = get_top_individuals(fitness, population)
        # print('Gen {}, Mutation Rate: {}, Top: {} '.format(gen, dm_rate, [x[:-1] for x in top_indvs]))
        for i in range(nw):
            history[i].append(top_indvs[i][:nw])
        for i in range(nw):
            if top_indvs[i][i] == last_fitness[i]:
                no_change_count[i] += 1
            else:
                no_change_count[i] =0
                dm_rate = m_rate
        if np.any(np.greater(no_change_count,2)):
            dm_rate = dm_rate*1.2
            if dm_rate > 0.15:
                dm_rate = 0.1*m_rate
            no_change_count = [0 for _ in range(nw)]
        for i in range(nw):
            last_fitness[i] = top_indvs[i][i]
        if all([x > 50 for x in no_change_count]):
            break
        # print('Mutation Rate: {}'.format(dm_rate))

        fitness, population = breed_and_cull(population, fitness, weights, population_size, keep_top_num, birthrate, m_rate)
    top_indvs = get_top_individuals(fitness, population)
    pop_history.append(fitness)
    return top_indvs[0], history, pop_history

def get_top_individuals(fitness, population):
    nw = len(fitness)
    pop_fit = list(zip(*fitness, population))
    top_indvs = []
    for i in range(nw):
        pop_fit.sort(reverse=True, key=lambda x: x[i])
        top_indvs.append(pop_fit[0])
    return top_indvs


def MultiRandMunkres(*weights,generations,population_size,birthrate=2.0,keep_top_num=1,m_rate=0.01):
    nw = len(weights)
    history = [[] for _ in range(nw)]
    pop_history = []
    fitness = [[] for _ in range(nw)]
    fitness[0], population = spawn_gen(population_size, weights[0])
    for i in range(1, nw):
        fitness[i] = [evaluate(p, weights[i]) for p in population]

    for gen in range(generations):
        pop_history.append(list(fitness))
        top_indvs = get_top_individuals(fitness, population)
        # print('Gen {}, Top: {}'.format(gen,top_indvs))
        for i in range(nw):
            history[i].append(top_indvs[i][:nw])
        fitness, population = random_breed(population, fitness, weights, population_size, keep_top_num, birthrate, m_rate)
    pop_fit1 = list(zip(*fitness, population))
    pop_fit1.sort(reverse=True, key=lambda x: x[0])
    top_indv = pop_fit1[0]
    pop_history.append(fitness)
    return top_indv, history, pop_history

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

def find_matching(weights1):
    nweights1 = np.max(weights1) - weights1
    m = Munkres()
    matching = m.compute(np.copy(nweights1))
    match_mat = np.zeros_like(weights1, dtype=bool)
    for j, r in matching:
        match_mat[j, r] = 1
    return match_mat

def gen_weights_and_plot(n=None,get_opt=True):
    if n is None:
        data = pickle.load(open('sparse_fitness.pckl', 'rb'))
        sweights1 = data['fitness']
        sweights2 = data['value']
        OM = data['OM']
    else:
        weights1 = np.random.random((n, n))
        weights2 = np.random.random((n, n))
        sweights1 = sparse_dict(0, 0)
        sweights2 = sparse_dict(0, 0)
        sweights1.fromarray(weights1)
        sweights2.fromarray(weights2)
    # weights2 = np.copy(weights1)
    if get_opt:
        weights1 = sweights1.todense()
        nweights1 = np.max(weights1) - weights1
        _, _, oM = find_and_eval_matching(nweights1)
        OM = sparse_dict(0, 0)
        OM.fromarray(oM)
        optimal1 = evaluate(OM, sweights1)
        optimal2 = evaluate(OM, sweights2)
    match, history, pop_history = MultiEvoMunkres(sweights1, sweights2, generations=600, population_size=30, birthrate=40.0, keep_top_num=3,m_rate=0.05)
    new_optimal1 = evaluate(match[-1], sweights1)
    new_optimal2 = evaluate(match[-1], sweights2)
<<<<<<< HEAD
    # prcnt_f1 = (new_optimal1-optimal1)/optimal1
    # prcnt_f2 = (new_optimal2-optimal2)/optimal2
    print(n,'New Optimal: {}, {}'.format(new_optimal1, new_optimal2))
    # print(n,'dF1: {}%, dF2: {}%'.format(prcnt_f1,prcnt_f2))
=======
    print(n, 'New Optimal: {}, {}'.format(new_optimal1, new_optimal2))
    if get_opt:
        print(n,'Old Optimal: {}, {}'.format(optimal1, optimal2))
        prcnt_f1 = (new_optimal1-optimal1)/optimal1
        prcnt_f2 = (new_optimal2-optimal2)/optimal2
        print(n, 'dF1: {}%, dF2: {}%'.format(prcnt_f1, prcnt_f2))


>>>>>>> 107caa0de61d7104d91745b506e5fc4c468a4b01

    h1, h2 = history
    h11, h12 = list(map(list, zip(*h1)))
    h21, h22 = list(map(list, zip(*h2)))
    x = list(range(len(h1)))
    plt.plot(x, h11, 'g--', label='Highest F1, F1')
    plt.plot(x, h12, 'g-.', label='Highest F1, F2')
    plt.plot(x, h21, 'm--', label='Highest F2, F1')
    plt.plot(x, h22, 'm-.', label='Highest F2, F2')
    if get_opt:
        plt.plot([0, len(h1)], [optimal1, optimal1], 'r--', label='Optimal F1')
        plt.plot([0, len(h1)], [optimal2, optimal2], 'b--', label='Sub-Optimal F2')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    # pickle.dump({'match': match, 'history': history, 'pop_history': pop_history}, open('history.pckl', 'wb'))
    if n is not None:
        plt.title('Vector Evaluated Genetic Algorithm, {}x{}'.format(n, n))
        plt.savefig('figure_out_{}x{}.png'.format(n, n))
    else:
        plt.title('Vector Evaluated Genetic Algorithm, GoGetter Data'.format(n, n))
        plt.savefig('figure_out_skills.png')


if __name__ == '__main__':
    import pickle
    n = 1000

    # p = Pool()
    # p.map(gen_weights_and_plot,[25,50,100,200,300,500,750,1000])
    gen_weights_and_plot(get_opt=False)








