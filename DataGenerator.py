import numpy as np


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




def gen_data(n_jobs=100,n_res=100,n_skills=256,n_filters=10):
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