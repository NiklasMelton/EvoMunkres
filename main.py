from munkres import Munkres
import pickle
import numpy as np
import matplotlib.pyplot as plt


def gen_resumes(n_res, n_skills, n_filters, skill_dist, filter_dist):
    resumes = [None]*n_res
    skills_mu = np.random.normal(skill_dist['mu']['mu'], skill_dist['mu']['sigma'], n_skills)
    filters_mu = np.random.normal(filter_dist['mu']['mu'], filter_dist['mu']['sigma'], n_filters)
    skills_sigma = np.random.normal(skill_dist['sigma']['mu'], skill_dist['sigma']['sigma'], n_skills)
    filters_sigma = np.random.normal(filter_dist['sigma']['mu'], filter_dist['sigma']['sigma'], n_filters)
    avg_n_skills=0
    avg_n_filters=0
    for i in range(n_res):
        resumes[i] = {'skills':np.zeros(0),'filters':np.zeros(0),'salary':0}
        while not np.any(resumes[i]['skills']):
            resumes[i]['skills'] = np.greater([np.random.normal(m,s) for m,s in zip(skills_mu,skills_sigma)],0.5)
        while not np.any(resumes[i]['filters']):
            resumes[i]['filters'] = np.greater([np.random.normal(m,s) for m,s in zip(filters_mu,filters_sigma)],0.5)
        avg_n_skills += np.sum(resumes[i]['skills'])
        avg_n_filters += np.sum(resumes[i]['filters'])
    avg_n_skills /= n_res
    avg_n_filters /= n_res
    print('{} skills and {} filters on average'.format(avg_n_skills,avg_n_filters))
    return resumes

def gen_jobs(n_jobs, n_skills, n_filters, skill_dist, filter_dist):
    jobs = [None]*n_jobs
    skills_mu = np.random.normal(skill_dist['mu']['mu'],skill_dist['mu']['sigma'],n_skills)
    filters_mu = np.random.normal(filter_dist['mu']['mu'],filter_dist['mu']['sigma'],n_filters)
    skills_sigma = np.random.normal(skill_dist['sigma']['mu'],skill_dist['sigma']['sigma'],n_skills)
    filters_sigma = np.random.normal(filter_dist['sigma']['mu'],filter_dist['sigma']['sigma'],n_filters)
    avg_n_skills=0
    avg_n_filters=0
    for i in range(n_jobs):
        jobs[i] = {'skills':np.zeros(0),'filters':np.zeros(0),'salary':0, 'fee':0}
        while not np.any(jobs[i]['skills']):
            jobs[i]['skills'] = np.greater([np.random.normal(m,s) for m,s in zip(skills_mu,skills_sigma)],0.5)
        while not np.any(jobs[i]['filters']):
            jobs[i]['filters'] = np.greater([np.random.normal(m,s) for m,s in zip(filters_mu,filters_sigma)],0.5)
        avg_n_skills += np.sum(jobs[i]['skills'])
        avg_n_filters += np.sum(jobs[i]['filters'])
    avg_n_skills /= n_jobs
    avg_n_filters /= n_jobs
    print('{} skills and {} filters on average'.format(avg_n_skills,avg_n_filters))
    return jobs

def gen_salaries(jobs,resumes,n_skills,skill_value_dist,salary_var_dist,fees,fee_freq):
    skill_rates = np.random.normal(skill_value_dist['mu'],skill_value_dist['sigma'],n_skills)
    for i in range(len(resumes)):
        salary = np.sum(np.multiply(resumes[i]['skills'],skill_rates))
        salary *= np.random.normal(salary_var_dist['mu'],salary_var_dist['sigma'])
        resumes[i]['salary'] = salary

    for i in range(len(jobs)):
        salary = np.sum(np.multiply(jobs[i]['skills'],skill_rates))
        salary *= np.random.normal(salary_var_dist['mu'],salary_var_dist['sigma'])
        jobs[i]['salary'] = salary
        jobs[i]['fee'] = np.random.choice(fees,p=fee_freq)
    return jobs, resumes


def jaccard(a,b):
    n = np.sum(np.minimum(a,b))
    d = np.sum(np.maximum(a,b))
    return n/d


def calculate_weights(jobs,resumes):
    n_jobs = len(jobs)
    n_res = len(resumes)
    fitness = np.zeros((n_jobs,n_res))
    value = np.zeros((n_jobs,n_res))
    salaries = np.zeros((n_jobs,n_res))
    print('Calculating weights')
    pass_count = 0
    total_count = 0
    for j, job in enumerate(jobs):
        # if not j%20:
        #     print('{} jobs complete'.format(j))
        j_filters = np.where(job['filters'])
        for r,res in enumerate(resumes):
            total_count += 1
            filter_pass = np.all(res['filters'][j_filters])
            if filter_pass:
                fitness[j,r] = jaccard(job['skills'],res['skills'])
                salaries[j,r] = (job['salary']+res['salary'])/2
                value[j,r] = job['fee']*(job['salary']+res['salary'])/2
                pass_count += 1
            else:
                fitness[j,r] = 0
                value[j,r] = 0
    print('Pass rate: {}/{} or {}'.format(pass_count,total_count,pass_count/total_count))
    return fitness, value, salaries




def gen_data():
    n_jobs = 100
    n_res = 100

    n_skills = 256
    n_filters = 10

    skill_dist = {
        'mu': {
            'mu': 0.4,
            'sigma': 0.12
        },
        'sigma': {
            'mu': 0.15,
            'sigma': 0.03
        }
    }

    job_skill_dist = {
        'mu': {
            'mu': 0.5,
            'sigma': 0.1
        },
        'sigma': {
            'mu': 0.15,
            'sigma': 0.03
        }
    }

    filter_dist = {
        'mu': {
            'mu': 0.8,
            'sigma': 0.12
        },
        'sigma': {
            'mu': 0.12,
            'sigma': 0.02
        }
    }

    job_filter_dist = {
        'mu': {
            'mu': 0.5,
            'sigma': 0.08
        },
        'sigma': {
            'mu': 0.12,
            'sigma': 0.02
        }
    }

    skill_value_dist = {
        'mu': 1,
        'sigma': 1
    }

    salary_var_dist = {
        'mu': 1,
        'sigma': 0.05
    }

    fees = [0.1, 0.125, 0.15, 0.175, 0.2]
    fee_freq = [0.3, 0.35, 0.2, 0.05, 0.1]


    resumes = gen_resumes(n_res, n_skills, n_filters, skill_dist, filter_dist)
    jobs = gen_jobs(n_jobs, n_skills, n_filters, job_skill_dist, job_filter_dist)
    jobs,resumes = gen_salaries(jobs,resumes,n_skills,skill_value_dist,salary_var_dist,fees,fee_freq)

    fitness, value, salaries  = calculate_weights(jobs,resumes)

    return fitness, value, salaries

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

def multi_obj_munkres(weights1,weights2,n=np.inf):
    weights1 = np.copy(weights1)
    weights2 = np.copy(weights2)
    weights2[weights2==1] = 0
    weights2 = 1-weights2
    w1_hist = []
    w2_hist = []
    iw1_hist = []
    iw2_hist = []
    e = 0
    n = np.minimum(n,np.minimum(weights1.shape[0],weights1.shape[1])-2)
    while e < n:
        matching, matching_weight1, matching_weight2, match_mat = find_and_eval_matching(np.copy(weights1),np.copy(weights2))
        w1_sum = np.sum(matching_weight1)
        w2_sum = np.sum(matching_weight2)
        # _, imatching_weight1, imatching_weight2, imatch_mat = find_and_eval_matching(np.copy(weights2),np.copy(weights1))
        # iw1_sum = np.sum(imatching_weight1)
        # iw2_sum = np.sum(imatching_weight2)
        iw1_sum = 0
        iw2_sum = 0
        w1_hist.append(w1_sum)
        w2_hist.append(w2_sum)
        # iw1_hist.append(iw1_sum)
        # iw2_hist.append(iw2_sum)
        print('W1: {}, W2: {}, iW1: {}, iW2: {}'.format(w1_sum,w2_sum,iw1_sum,iw2_sum))
        # common_match = np.logical_and(match_mat,imatch_mat)
        pt_found = False
        if False:#np.any(common_match):
            print('PT A')
            common_match_pts= np.where(common_match)
            for pi,pj in zip(*common_match_pts):
                if np.sum(weights1[pi, :]) < weights1.shape[1] - 1 and np.sum(weights1[:, pj]) < weights1.shape[0] - 1:
                    weights1[pi, pj] = 1
                    weights2[pi, pj] = 1
                    pt_found = True
        if not pt_found:
            print('PT B')
            flat_match = match_mat.flatten()
            match_mat_idx = np.where(match_mat)
            match_points = weights2.flatten()[flat_match]
            match_points_sorted_idxs = np.argsort(match_points)

            p = 0
            while not pt_found and p < len(match_points_sorted_idxs):
                rem_pt = match_points_sorted_idxs[p]
                pi = match_mat_idx[0][rem_pt]
                pj = match_mat_idx[1][rem_pt]
                if np.sum(weights1[pi,:]) < weights1.shape[1]-1 and np.sum(weights1[:,pj]) < weights1.shape[0]-1:
                    weights1[pi,pj] = 1
                    weights2[pi,pj] = 1

                    if p > 10:
                        pt_found = True
                p += 1
            if not pt_found:
                break
        e += 1

    return matching, w1_hist, w2_hist, iw1_hist, iw2_hist


def val2std(mat):
    smat = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            d = mat[i,:].tolist() + mat[:,j].tolist()
            del d[j]
            m = np.mean(d)
            s = np.std(d)
            smat[i,j] = (mat[i,j]-m)/s
    return smat


def multi_obj_munkres2(weights1,weights2):
    weights1 = np.copy(weights1)
    weights2 = np.copy(weights2)
    sw1 = val2std(weights1)
    sw2 = val2std(weights2)

    print(np.max(sw1))
    print(np.min(sw1))
    print(np.max(sw2))
    print(np.min(sw2))

    avg_sw1 = sw1 > -0.5
    high_sw2 = sw2 > -0.5

    remove_pts = np.logical_and(avg_sw1, high_sw2)
    print('removing {} edges'.format(np.sum(remove_pts)))

    matching, matching_weight1, matching_weight2, match_mat = find_and_eval_matching(np.copy(weights1),np.copy(weights2))
    ow1_sum = np.sum(matching_weight1)
    ow2_sum = np.sum(matching_weight2)
    print('Origional: w1: {},  w2: {}'.format(ow1_sum, ow2_sum))

    remove_pts = np.logical_and(avg_sw1, high_sw2)
    print('removing {} common edges'.format(np.sum(np.logical_and(remove_pts,match_mat))))


    weights1[remove_pts] = 1
    weights2[remove_pts] = 1

    matching, matching_weight1, matching_weight2, match_mat = find_and_eval_matching(np.copy(weights1),np.copy(weights2))
    w1_sum = np.sum(matching_weight1)
    w2_sum = np.sum(matching_weight2)


    print('New: w1: {},  w2: {}'.format(w1_sum,w2_sum))

    return matching


if __name__ =='__main__':
    np.random.seed(11111)
    fitness,value,salaries = gen_data()
    print('Plotting')
    print('Fitness: Min: {}, Max: {}, Mean: {}'.format(np.min(fitness),np.max(fitness),np.mean(fitness)))
    print('Value: Min: {}, Max: {}, Mean: {}'.format(np.min(value),np.max(value),np.mean(value)))
    print('Salary: Min: {}, Max: {}, Mean: {}'.format(np.min(salaries),np.max(salaries),np.mean(salaries)))
    mf = np.max(fitness)
    print(mf)
    print(sum(sum(fitness==mf)))
    disfitness = 1 - fitness/np.max(fitness)
    print(sum(sum(disfitness == 0)))
    disvalue = 1 - value/np.max(value)

    # matching = m.compute(disfitness)
    # # print('Matching computed')
    # matching_fitness = []
    # matching_value = []
    # value_match = np.zeros_like(fitness, dtype=bool)
    # for j, r in matching:
    #     matching_fitness.append(fitness[j, r])
    #     matching_value.append(value[j, r])
    #     value_match[j, r] = 1
    #
    # print('Total Fitness: {},  Total Value: {}'.format(sum(matching_fitness), sum(matching_value)))
    #
    # match_same = np.sum(np.logical_and(fitness_match, value_match))
    # match_diff = np.sum(np.logical_xor(fitness_match, value_match))
    #
    # print('{} common matchings, {} different'.format(match_same, match_diff))

    # matching, w1_hist, w2_hist, iw1_hist, iw2_hist = multi_obj_munkres(disfitness, disvalue,n=100)
    matching = multi_obj_munkres2(disfitness, disvalue)

    # plt.figure()
    # x = list(range(len(w1_hist)))
    # plt.plot(x,w1_hist,'g-',x,w2_hist,'r-')
    # plt.show()



