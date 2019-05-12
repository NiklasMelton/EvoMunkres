import json
import numpy as np
import pickle
from scipy.sparse import coo_matrix
from multiprocessing import Pool

def get_skill_recursive(tax,lbls):
    lbl = lbls[0]
    skills = []
    try:
        if lbl is None:
            sub_tax_list = tax
        elif lbl in tax:
            sub_tax_list = tax[lbl]
        else:
            return skills
        for sub_tax in sub_tax_list:
            skills.append(sub_tax['@name'])
            if len(lbls) > 1:
                skills += get_skill_recursive(sub_tax,lbls[1:])
    except Exception as E:
        print('Recursive Skill Error:',E)
    return skills

def extract_tax(doc,dtype):
    tax = []
    if dtype == 'resume':
        res = doc['Resume']['UserArea']['sov:ResumeUserArea']
        smry = res['sov:ExperienceSummary']
        if 'sov:SkillsTaxonomyOutput' in smry:
            root = smry['sov:SkillsTaxonomyOutput']['sov:TaxonomyRoot']
            if len(root) > 0:
                tax = root[0]['sov:Taxonomy']
    else:
        if 'SkillsTaxonomyOutput' in doc:
            root = doc['SkillsTaxonomyOutput']
            if len(root) > 0:
                tax = root[0]['Taxonomy']
    return tax


def get_skills(doc,dtype):
    skills = []
    lbls = [None, 'Subtaxonomy', 'Skill', 'ChildSkill']
    if dtype == 'resume':
        lbls = [x if x is None else 'sov:' + x for x in lbls]
    try:
        tax = extract_tax(doc,dtype)
        if tax:
            skills = get_skill_recursive(tax,lbls)
    except Exception as E:
        print('Skill Error:', E)
        exit()
    return set(skills)

def ann2hour(rate):
    return float(rate)/(2080.)

def get_payrate(job):
    val = job['PayRate']['@amount']
    unit = job['PayRate']['@unit']
    if unit == 'year':
        val = ann2hour(val)
    else:
        val = float(val)
    return val

def extract_job_requirements(doc):
    reqs = {}
    if 'CertificationsAndLicenses' in doc:
        reqs['certs'] = list(doc['CertificationsAndLicenses']['CertificationOrLicense'])
    else:
        reqs['certs'] = []
    if 'RequiredSkills' in doc:
        reqs['skills'] = list(doc['RequiredSkills']['RequiredSkill'])
    else:
        reqs['skills'] = []
    # reqs['loc'] = dict(doc['CurrentLocation'])
    if 'MinimumYears' in doc:
        reqs['min_years'] = float(doc['MinimumYears'])
    else:
        reqs['min_years'] = 0.
    if 'MaximumYears' in doc:
        reqs['max_years'] = float(doc['MaximumYears'])
    else:
        reqs['max_years'] = 100.
    if 'MaximumYearsManagement' in doc:
        reqs['max_years_mgmt'] = float(doc['MaximumYearsManagement'])
    else:
        reqs['max_years_mgmt'] = 100.
    if 'MinimumYearsManagement' in doc:
        reqs['min_years_mgmt'] = float(doc['MinimumYearsManagement'])
    else:
        reqs['min_years_mgmt'] = 0.
    if 'HighestManagementScore' in doc:
        reqs['mgmt_score'] = float(doc['HighestManagementScore'])
    else:
        reqs['mgmt_score'] = 0
    if 'ExecutiveType' in doc and doc['ExecutiveType'] != 'NONE':
        reqs['exec_type'] = doc['ExecutiveType']
    if 'RequiredDegree' in doc and doc['RequiredDegree'] != 'UNSPECIFIED' and doc['RequiredDegree'] != 'NONE':
        reqs['degrees'] = doc['RequiredDegree']
    if 'JobTitles' in doc:
        reqs['titles'] = doc['JobTitles']['MainJobTitle']
    # reqs['crnt_mgmt'] = bool(doc('CurrentJobIsMgmt'))
    return reqs

def extract_res_qualifications(doc):
    quals = {}
    if 'sov:ExperienceSummary' in doc['Resume']:
        if 'sov:MonthsOfWorkExperience' in doc['Resume']['sov:ExperienceSummary']:
            quals['years_exp'] = float(doc['Resume']['sov:ExperienceSummary']['sov:MonthsOfWorkExperience'])/12
        else:
            quals['years_exp'] = 0
        if 'sov:MonthsOfManagementExperience' in doc['Resume']['sov:ExperienceSummary']:
            quals['years_mgmt'] = float(doc['Resume']['sov:ExperienceSummary']['sov:MonthsOfManagementExperience'])/12
        else:
            quals['years_mgmt'] = 0
        if 'sov:HighestManagementScore' in doc['Resume']['sov:ExperienceSummary']:
            quals['mgmt_score'] = float(doc['Resume']['sov:ExperienceSummary']['sov:HighestManagementScore'])
        else:
            quals['mgmt_score'] = 0
        if 'sov:ExecutiveType' in doc['Resume']['sov:ExperienceSummary']:
            quals['exec_type'] = doc['Resume']['sov:ExperienceSummary']['sov:ExecutiveType']
        else:
            quals['exec_type'] = 'NONE'
    else:
        quals['years_exp'] = 0
        quals['years_mgmt'] = 0
        quals['mgmt_score'] = 0
    # quals['loc'] = doc['sov:Location']

    if 'LicensesAndCertifications' in doc['Resume']['StructuredXMLResume']:
        quals['certs'] = [x['Name'] for x in doc['Resume']['StructuredXMLResume']['LicensesAndCertifications']['LicenseOrCertification']]
    else:
        quals['certs'] = []
    if 'EducationHistory' in doc['Resume']['StructuredXMLResume']:
        quals['degrees'] = list(set([y['@degreeType'] for x in doc['Resume']['StructuredXMLResume']['EducationHistory']['SchoolOrInstitution'] for y in x['Degree'] if '@degreeType' in y ]))
    else:
        quals['degrees'] = []
    if 'EmploymentHistory' in doc['Resume']['StructuredXMLResume']:
        ttls = []
        for x in doc['Resume']['StructuredXMLResume']['EmploymentHistory']['EmployerOrg']:
            if 'PositionHistory' in x:
                for y in x['PositionHistory']:
                    if 'Title' in y:
                        ttls.append(y['Title'])
        quals['titles'] = list(set(ttls))
    else:
        quals['titles'] = []

    quals['skills'] = get_skills(doc,'resume')
    return quals

DEGREE_RANK = {
    'lessThanHighSchool': -1,
    'some high school or equivalent': -1,
    'secondary': 0,
    'ged': 0,
    'highSchoolOrEquivalent': 0,
    'high school or equivalent': 0,
    'someCollege': 0.5,
    'some college': 0.5,
    'certification': 0.75,
    'associates': 1,
    'HND/HNC or equivalent': 1,
    'bachelors': 2,
    'somePostgraduate': 2.5,
    'some post-graduate': 2.5,
    'masters': 3,
    'intermediategraduate': 3.5,
    'professional': 4,
    'vocational': 4,
    'doctorate': 4
}

DEGREE_NORMS = {
    'lessThanHighSchool': 'lessThanHighSchool',
    'some high school or equivalent': 'lessThanHighSchool',
    'secondary': 'secondary',
    'ged': 'ged',
    'highSchoolOrEquivalent': 'highSchoolOrEquivalent',
    'high school or equivalent': 'highSchoolOrEquivalent',
    'someCollege': 'someCollege',
    'some college': 'someCollege',
    'certification': 'certification',
    'associates': 'associates',
    'HND/HNC or equivalent': 'HND/HNC or equivalent',
    'bachelors': 'bachelors',
    'somePostgraduate': 'somePostgraduate',
    'some post-graduate': 'somePostgraduate',
    'masters': 'masters',
    'intermediategraduate': 'intermediategraduate',
    'professional': 'professional',
    'vocational': 'vocational',
    'doctorate': 'doctorate'
}

MANDATORY_DEGREES = ['vocational','professional','doctorate','certification']

def degrees_superior(req_deg,qual_degs):
    if req_deg not in DEGREE_NORMS:
        raise ValueError('Missing {} in Degree Norms'.format(req_deg))
    if not all([deg in DEGREE_NORMS for deg in qual_degs]):
        raise ValueError('Missing {} in Degree NORMS'.format(qual_degs))
    nreq_deg = DEGREE_NORMS[req_deg]
    nqual_degs = [DEGREE_NORMS[deg] for deg in qual_degs]
    if nreq_deg and not nqual_degs:
        return False
    if nreq_deg in nqual_degs:
        return True
    if nreq_deg in MANDATORY_DEGREES:
        return False
    req_rank = DEGREE_RANK[nreq_deg]
    qual_rank = max([DEGREE_RANK[deg] for deg in nqual_degs])
    return qual_rank > req_rank

def qualify_resume_for_job(reqs,quals):
    if 'degrees' in reqs and not degrees_superior(reqs['degrees'],quals['degrees']):
        # print('degrees',reqs['degrees'],quals['degrees'])
        return False
    if 'skills' in reqs and reqs['skills'] and (not quals['skills'] or len(set(reqs['skills']).intersection(quals['skills'])) < len(reqs['skills'])):
        # print('skills')#,reqs['skills'],len(quals['skills']))
        return False
    if 'certs' in reqs and not all([c in quals['certs'] for c in reqs['certs']]):
        # print('certs')
        return False
    if not reqs['min_years'] <= quals['years_exp'] <= reqs['max_years']:
        # print('years')
        return False
    if not reqs['min_years_mgmt'] <= quals['years_mgmt'] <= reqs['max_years_mgmt']:
        # print('mgmt years')
        return False
    if 'titles' in reqs and not any([t in reqs['titles'] for t in quals['titles']]):
        # print('titles')
        return False
    if quals['mgmt_score'] < reqs['mgmt_score']-10:
        # print('mgmt score')
        return False
    # print('------------QUALIFIED-----------')
    return True


def score_resume_for_job(job_skills,quals):
    js = set(job_skills)
    qs = set(quals['skills'])
    if not js:
        # print('No Job skills')
        return 0.1
    return float(len(js.intersection(qs)))/float(len(js))



def eval_res_pairs(resume,job_skill_reqs=None,jobs=None,r=0):
    if job_skill_reqs is None and jobs is None:
        raise ValueError('Must provide jobs or skills')
    if job_skill_reqs is None:
        nj = len(jobs)
        job_skills = []
        job_reqs = []
        fp = True
    else:
        job_skills = job_skill_reqs['skills']
        job_reqs = job_skill_reqs['reqs']
        nj = len(job_skills)
        fp = False
    res_doc = json.loads(json.loads(resume)['Value']['ParsedDocument'])
    res_quals = extract_res_qualifications(res_doc)
    skill_fitness_data = []
    skill_fitness_row = []
    skill_fitness_col = []
    for j in range(nj):
        if fp:
            job_doc = json.loads(json.loads(jobs[j])['Value']['ParsedDocument'])['SovrenData']
            job_reqs.append(extract_job_requirements(job_doc))
            job_skills.append(get_skills(job_doc, 'job').union(job_reqs[j]['skills']))
        if qualify_resume_for_job(job_reqs[j], res_quals):
            similarity = score_resume_for_job(job_skills[j], res_quals)
            skill_fitness_data.append(similarity)
            skill_fitness_row.append(r)
            skill_fitness_col.append(j)
    if fp:
        return (skill_fitness_data,skill_fitness_row, skill_fitness_col), {'skills': job_skills, 'reqs': job_reqs}
    else:
        return (skill_fitness_data,skill_fitness_row, skill_fitness_col)

def par_eval_res_pairs(args):
    return eval_res_pairs(resume = args[0],job_skill_reqs=args[1],r=args[2])


if __name__ == '__main__':
    sovren_data = pickle.load(open('../cache/sovren_cache_dump.pckl','rb'))

    resumes = sovren_data['res']
    jobs = sovren_data['job']

    nr = len(resumes)
    nj = len(jobs)
    print('{} resumes, {} jobs'.format(nr,nj))

    fitness_data, job_skill_reqs = eval_res_pairs(resumes[0],jobs=jobs)
    res_pairs = [(res,dict(job_skill_reqs),nr+1) for nr, res in
 enumerate(resumes[1:])]
    P = Pool()
    par_fitness_data = P.map(par_eval_res_pairs,res_pairs)
    # par_fitness_data = [par_eval_res_pairs(res_pair) for res_pair in res_pairs]
    # for res_pair in res_pairs:
    #     data = par_eval_res_pairs(res_pair)
    #     print([len(d) for d in data])
    fitness_data = list(fitness_data)
    for par_eval in par_fitness_data:
        fitness_data[0] += par_eval[0]
        fitness_data[1] += par_eval[1]
        fitness_data[2] += par_eval[2]
    print(len(fitness_data)/(nj*nr))
    pickle.dump(fitness_data,open('skill_fitness.pckl','wb'))
