from munkres import Munkres

def evaluate(matching, weights):
    return weights[matching].sum()

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
    import pickle
    import numpy as np
    from sparse_dict import sparse_dict

    data = pickle.load(open('sparse_fitness.pckl','rb'))
    sweights1 = data['fitness']
    sweights2 = data['value']
    weights1 = sweights1.todense()

    nweights1 = np.max(weights1) - weights1

    x,y = nweights1.shape
    n = max(x,y)
    nweights1 = np.pad(nweights1,([0,n-x],[0,n-y]),'constant',constant_values=0)
    print(nweights1.shape)

    _,_,oM = find_and_eval_matching(nweights1)
    OM = sparse_dict(0,0)
    OM.fromarray(oM)
    optimal1 = evaluate(OM,sweights1)
    optimal2 = evaluate(OM,sweights2)
    print('Optimal: {}, {}'.format(optimal1, optimal2))
    pickle.dump({'fitness': sweights1, 'value': sweights2,'OM':OM}, open('sparse_fitness.pckl', 'wb'))