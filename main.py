from munkres import Munkres
import pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
# from DataGenerator import  gen_data
from SparseMultiEvoMunkres import MultiEvoMunkres, MultiRandMunkres
from SparseYaoEvo import YaoEvo
from SparseSimulatedAnnealing import MultiAnneal, ScalarMultiAnneal
from ModMunkres import ModMunkres


def find_matching(weights1):
    nweights1 = np.max(weights1) - weights1
    m = Munkres()
    matching = m.compute(np.copy(nweights1))
    match_mat = np.zeros_like(weights1, dtype=bool)
    for j, r in matching:
        match_mat[j, r] = 1
    return match_mat

def compare_pareto_space(*weights):
    _, _, rand_hist = MultiRandMunkres(*weights, generations=300, population_size=30,birthrate=30.0, keep_top_num=10, m_rate=0.05)
    _, _, evo_hist = MultiEvoMunkres(*weights,generations=300,population_size=30,birthrate=40.0,keep_top_num=3,m_rate=0.05)
    _, _, yao_hist = YaoEvo(*weights,generations=300,population_size=30)
    _, _, anneal_hist = MultiAnneal(*weights,population_size=30,t0=100, tr=0.003,m_rate=0.15)
    _, _, munkres_A_hist = ModMunkres(*weights)
    _, _, munkres_B_hist = ModMunkres(*weights[::-1])

    rand_hist = [(x,y) for a in rand_hist for x,y in zip(*a)]
    evo_hist = [(x,y) for a in evo_hist for x,y in zip(*a)]
    yao_hist = [(x,y) for a in yao_hist for x,y in zip(*a)]
    anneal_hist = [(x,y) for a in anneal_hist for x,y in zip(*a)]
    vanneal_hist = [(x,y) for a in vanneal_hist for x,y in zip(*a)]



    rand_hist = list(map(list,zip(*rand_hist)))

    evo_hist = list(map(list,zip(*evo_hist)))

    yao_hist = list(map(list,zip(*yao_hist)))

    anneal_hist = list(map(list,zip(*anneal_hist)))
    vanneal_hist = list(map(list,zip(*vanneal_hist)))

    munkres_A_hist = list(map(list,zip(*munkres_A_hist)))
    munkres_B_hist = list(map(list,zip(*munkres_B_hist)))
    munkres_B_hist = munkres_B_hist[::-1]

    plt.plot(*anneal_hist, 'k.')
    plt.plot(*vanneal_hist, 'm.')
    plt.plot(*rand_hist, 'r.')
    plt.plot(*evo_hist, 'b.')
    plt.plot(*yao_hist, 'g.')
    plt.plot(*munkres_A_hist, 'c.')
    plt.plot(*munkres_B_hist, 'y.')

    plt.xlabel('Fitness Function 1')
    plt.ylabel('Fitness Function 2')
    plt.legend(['Scalar Annealing','Vector Annealing', 'Random','Evolutionary 1', 'Evolutionary 2', 'Modified Munkres A','Modified Munkres B'])
    plt.title('Fitness Space')
    plt.show()


def compare_iterations(*weights):
    _, rand_hist, _ = MultiRandMunkres(*weights, generations=625, population_size=30, birthrate=30.0, keep_top_num=10,
                                       m_rate=0.05)
    _, evo_hist, _ = MultiEvoMunkres(*weights, generations=625, population_size=30, birthrate=40.0, keep_top_num=3,
                                     m_rate=0.05)
    _, yao_hist, _ = YaoEvo(*weights, generations=625, population_size=30)
    _, vanneal_hist, _ = MultiAnneal(*weights, population_size=30, t0=10000, tr=0.03, m_rate=0.15)
    _, anneal_hist, _ = ScalarMultiAnneal(*weights, population_size=30, t0=10000, tr=0.03, m_rate=0.15)
    _, munkres_A_hist, _ = ModMunkres(*weights)
    _, munkres_B_hist, _ = ModMunkres(*weights[::-1])

    # hists = [rand_hist,evo_hist]
    hists = [rand_hist,evo_hist,yao_hist,anneal_hist, vanneal_hist]
    colors = ['r','b','g','k','m']
    labels = ['Random','Evolutionary 1', 'Evolutionary 2', 'Scalar Annealing','Vector Annealing']
    for i,(f1,f2) in enumerate(hists):
        f11, f12 = list(map(list,zip(*f1)))
        f21, f22 = list(map(list,zip(*f2)))
        x = list(range(len(f11)))
        plt.plot(x,f11,colors[i]+'-',label=labels[i])
        plt.plot(x,f12,colors[i]+'--',label=None)
        plt.plot(x,f22,colors[i]+'-.',label=None)
        plt.plot(x,f22,colors[i]+':',label=None)
    f1, f2 = munkres_A_hist
    x = list(range(len(f1)))
    plt.plot(x,f1,'c-',label='Modified Munkres A')
    plt.plot(x,f2,'c--', label=None)
    f1, f2 = munkres_B_hist
    plt.plot(x,f1,'y-',label='Modified Munkres B')
    plt.plot(x,f2,'y--',label=None)
    plt.title('Fitness over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()


def pcompare(args):
    weights, fname = args
    if fname == 'MRM':
        return MultiRandMunkres(*weights, generations=625, population_size=30, birthrate=30.0, keep_top_num=10,m_rate=0.05)
    elif fname == 'MEM':
        return MultiEvoMunkres(*weights, generations=625, population_size=30, birthrate=40.0, keep_top_num=3,m_rate=0.05)
    elif fname == 'YE':
        return YaoEvo(*weights, generations=625, population_size=30)
    elif fname == 'MA':
        return MultiAnneal(*weights, population_size=30, t0=10000, tr=0.03, m_rate=0.15)
    elif fname == 'SMA':
        return ScalarMultiAnneal(*weights, population_size=30, t0=10000, tr=0.03, m_rate=0.15)
    # elif fname == 'MM':
    #     return ModMunkres(*weights)


def compare(*weights):

    # fnames = ['MRM','MEM','YE','SMA','MA','MM']
    fnames = ['MRM','MEM','YE','SMA','MA']
    args = [(weights,fname) for fname in fnames]
    # args.append((weights[::-1],'MM'))
    p = Pool()
    # data = p.map(pcompare,args)
    data = [pcompare(a) for a in args]
    # [[_, rand_hist, rand_hist1], [_, evo_hist, evo_hist1],[_, yao_hist, yao_hist1],[_, vanneal_hist, vanneal_hist1],
    #  [_, anneal_hist, anneal_hist1],[_, munkres_A_hist, munkres_A_hist1], [_, munkres_B_hist, munkres_B_hist1]] = data = p.map(pcompare,args)
    [[_, rand_hist, rand_hist1], [_, evo_hist, evo_hist1],[_, yao_hist, yao_hist1],[_, vanneal_hist, vanneal_hist1],
     [_, anneal_hist, anneal_hist1]] = data
    pickle.dump(data,open('sparse_output.pckl', 'wb'))
    # [[_, rand_hist, rand_hist1], [_, evo_hist, evo_hist1], [_, yao_hist, yao_hist1], [_, vanneal_hist, vanneal_hist1],
    #  [_, anneal_hist, anneal_hist1], [_, munkres_A_hist, munkres_A_hist1], [_, munkres_B_hist, munkres_B_hist1]] = pickle.load(open('output.pckl','rb'))

    print(len(rand_hist),len(anneal_hist))
    print(len(rand_hist[0]),len(anneal_hist[0]))
    print(len(rand_hist[0][0]),len(anneal_hist[0][0]))
    print(rand_hist[0][0],anneal_hist[0][0])

# hists = [rand_hist,evo_hist]
    hists = [rand_hist, evo_hist, yao_hist, anneal_hist, vanneal_hist]
    labels = ['Random', 'Evolutionary 1','Evolutionary 2', 'Scalar Annealing', 'Vector Annealing']
    colors = ['r', 'b','g', 'k', 'm']
    for i, (f1, f2) in enumerate(hists):
        f11, f12 = list(map(list, zip(*f1)))
        f21, f22 = list(map(list, zip(*f2)))
        x = list(range(len(f11)))
        plt.plot(x, f11, colors[i] + '-', label=labels[i])
        plt.plot(x, f12, colors[i] + '--', label=None)
        plt.plot(x, f21, colors[i] + '-.', label=None)
        plt.plot(x, f22, colors[i] + ':', label=None)
    # f1, f2 = munkres_A_hist
    # x = list(range(len(f1)))
    # plt.plot(x, f1, 'c-', label='Modified Munkres A')
    # plt.plot(x, f2, 'c--', label=None)
    # f1, f2 = munkres_B_hist
    # plt.plot(x, f1, 'y-', label='Modified Munkres B')
    # plt.plot(x, f2, 'y--', label=None)
    plt.title('Fitness over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()

    plt.figure()
    rand_hist = [(x, y) for a in rand_hist1 for x, y in zip(*a)]
    evo_hist = [(x, y) for a in evo_hist1 for x, y in zip(*a)]
    yao_hist = [(x, y) for a in yao_hist1 for x, y in zip(*a)]
    anneal_hist = [(x, y) for a in anneal_hist1 for x, y in zip(*a)]
    vanneal_hist = [(x, y) for a in vanneal_hist1 for x, y in zip(*a)]

    rand_hist = list(map(list, zip(*rand_hist)))

    evo_hist = list(map(list, zip(*evo_hist)))

    yao_hist = list(map(list, zip(*yao_hist)))

    anneal_hist = list(map(list, zip(*anneal_hist)))
    vanneal_hist = list(map(list, zip(*vanneal_hist)))

    # munkres_A_hist = list(map(list, zip(*munkres_A_hist1)))
    # munkres_B_hist = list(map(list, zip(*munkres_B_hist1)))
    # munkres_B_hist = munkres_B_hist[::-1]
    mx = 15
    plt.plot(*rand_hist, 'r.', markersize=mx)
    plt.plot(*vanneal_hist, 'm.', markersize=mx)
    plt.plot(*anneal_hist, 'k.',markersize=mx)


    plt.plot(*evo_hist, 'b.',markersize=mx)
    plt.plot(*yao_hist, 'g.',markersize=mx)
    # plt.plot(*munkres_A_hist, 'c.',markersize=mx)
    # plt.plot(*munkres_B_hist, 'y.',markersize=mx)

    plt.xlabel('Fitness Function 1')
    plt.ylabel('Fitness Function 2')
    # plt.legend(
    #     ['Random', 'Vector Annealing','Scalar Annealing', 'Evolutionary 1', 'Evolutionary 2', 'Modified Munkres A',
    #      'Modified Munkres B'])
    plt.legend(
        ['Random', 'Vector Annealing','Scalar Annealing', 'Evolutionary 1', 'Evolutionary 2'])
    plt.title('Fitness Space')
    plt.show()

if __name__ =='__main__':
    np.random.seed(11111)
    weights1 = np.random.random((25,25))
    weights2 = np.random.random((25,25))
    # compare_pareto_space(weights1,weights2)
    compare(weights1,weights2)


