import pickle
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import warnings
from collections import Counter
import sys

# Ignore numpy 1.15 FutureWarning for theano
warnings.filterwarnings('ignore')

# Constants preparation -------------------------------------------------------

COND_NUM = sys.argv[1]
CONDITION_FILE = 'cond_{}.bpkl3'.format(COND_NUM)
EPS = np.finfo(float).eps
OBS_MIN = 0
NUM_WEEKS = 570
PERIOD = 365.25 / 7.
NUM_BASES = 5
NUM_CHAINS = 1

AGE_GROUPS = {'0-10': 0,
              '11-20': 1,
              '21-30': 2,
              '31-40': 3,
              '41-50': 4,
              '51-65': 5}

AGE_GROUP_NUM = 6

# -----------------------------------------------------------------------------
            

def count2array(count, length, first):
    
    return np.array([count[i+1+first] for i in range(length)], dtype=float)
    
    
def plotRawTrend(sample, nw=NUM_WEEKS, alpha=1., ax=None):
    
    rawObsCount = Counter()
    rawTotCount  = Counter()
    
    for key, val in sample.items():
        
        rawObsCount  += val['RealObservation']
        rawTotCount  += val['PopSize']
    
    rawObs = np.array([rawObsCount[i+1] for i in range(NUM_WEEKS)])
    rawTot = np.array([rawTotCount[i+1] for i in range(NUM_WEEKS)])
    
    if ax is not None:
        ax.plot(rawObs / rawTot, alpha=alpha, 
                label='Raw Trend')
    else:
        plt.plot(rawObs / rawTot, alpha=alpha, 
                label='Raw Trend')
    
    return rawObs, rawTot


def sample2Inputs(sample):
    
    flatX = np.array([], dtype=float)
    flatPopIndex = np.array([], dtype=int)
    flatPopSize = np.array([], dtype=float)
    flatObservation = np.array([], dtype=float)
    flatAgeIndex = np.array([], dtype=int)
    
    allKeys = sorted(list(sample.keys()))
    outKeys = []
    
    i = 0
    for handle in allKeys:
        
        start = handle[0] - 1
        end = handle[1] 
        l = end - start
        
        trendCounts = sample[handle]
        obs =  count2array(trendCounts['HolidaySmoothObservation'], l, start)
        tot =  count2array(trendCounts['PopSize'], l,  start)
        
        if (len(np.nonzero(obs)[0]) >= OBS_MIN):
            
            ageIndex = AGE_GROUPS[handle[3]]
            
            indices_pops = np.repeat(i, end - start)
            flatPopIndex = np.concatenate((flatPopIndex, indices_pops))
            
            weeks = np.arange(start, end) + 1
            flatX = np.concatenate((flatX, weeks))
            
            flatPopSize = np.concatenate((flatPopSize, tot))
            
            flatObservation = np.concatenate((flatObservation, obs))
            
            flatAgeIndex = np.append(flatAgeIndex, ageIndex)
            
            outKeys.append(handle)
            i += 1
    
    return (flatX, flatPopIndex, 
            flatPopSize, flatObservation, 
            flatAgeIndex, outKeys)


def mcmc_Model(N, breakpoints, ageBarsNum, flatAgeIndex, 
               flatX, flatPopIndex, flatPopSize, flatObservation):
    
    def clip0to(rv):
        
        return tt.clip(rv, EPS, rv)
                                     
    n_bp = len(breakpoints)
    xIndex = np.int_(flatX - 1)
    breakpointsIndMat = np.zeros((NUM_WEEKS, n_bp))
    
    ageBars = np.arange(ageBarsNum)
    ageBars_X = ageBars[:, np.newaxis]
    
    flatProportion = flatObservation / flatPopSize
        
    for i in range(n_bp):
        zeroPad = breakpoints[i]
        breakpointsIndMat[:, i] = np.repeat([0,1], [zeroPad, NUM_WEEKS-zeroPad])
        
    periodic_bases1 = np.array([[np.cos(2.*np.pi*i*t / PERIOD) 
                                 for i in np.arange(1, NUM_BASES+1)] 
                                for t in np.arange(1, NUM_WEEKS+1)])
    
    periodic_bases2 = np.array([[np.sin(2.*np.pi*i*t / PERIOD) 
                                 for i in np.arange(1, NUM_BASES+1)] 
                                for t in np.arange(1, NUM_WEEKS+1)])
    
    periodic_bases = np.hstack((periodic_bases1, periodic_bases2))
        
    basic_model = pm.Model()
    
    with basic_model:

        # Hyperpriors for unknown model parameters
        l_α0 = pm.InverseGamma('l_α0', alpha=1., beta=1.)
        s2_α0 = pm.HalfCauchy('s2_α0', 5.)
        cov_α0 = s2_α0 * pm.gp.cov.ExpQuad(1, l_α0)
        gp_α0 = pm.gp.Latent(cov_func=cov_α0)
        α0 = gp_α0.prior('α0', X=ageBars_X)
        
        l_β0 = pm.InverseGamma('l_β0', alpha=1., beta=1.)
        s2_β0 = pm.HalfCauchy('s2_β0', 5.)
        cov_β0 = s2_β0 * pm.gp.cov.ExpQuad(1, l_β0)
        gp_β0 = pm.gp.Latent(cov_func=cov_β0)
        β0 = gp_β0.prior('β0', X=ageBars_X)
        
        σ_α = pm.HalfCauchy('σ_α', 5., shape=ageBarsNum)
        σ_β = pm.HalfCauchy('σ_β', 5., shape=ageBarsNum)
        
        skew_α_scale = pm.HalfCauchy('skew_α_scale ', 5.)
        skew_α = pm.Laplace('skew_α', mu=0., b=clip0to(skew_α_scale), 
                            shape=ageBarsNum)
        
        skew_β_scale = pm.HalfCauchy('skew_β_scale ', 5.)
        skew_β = pm.Laplace('skew_β', mu=0., b=clip0to(skew_β_scale), 
                            shape=ageBarsNum)
        
        # Shifts at the beginning of every year
        ss = pm.HalfCauchy('ss', 5.)
        shifts = pm.Laplace('shifts', mu=0., b=clip0to(ss), 
                            shape=(n_bp, N))
        
        # Periodic effects
        ps = pm.HalfCauchy('ps', 5.)
        periodics = pm.Normal('periodics', mu=0., sd=clip0to(ps), 
                            shape=(2*NUM_BASES, N))
        
        # Priors   
        α = pm.SkewNormal('α', 
                          mu=α0[flatAgeIndex],
                          sd=clip0to(σ_α[flatAgeIndex]),
                          alpha=skew_α[flatAgeIndex],
                          shape=N)     
        
        β = pm.SkewNormal('β', 
                          mu=β0[flatAgeIndex],
                          sd=clip0to(σ_β[flatAgeIndex]),
                          alpha=skew_β[flatAgeIndex],
                          shape=N)
        
        linear = ( α[flatPopIndex] +
                   β[flatPopIndex] * flatX +
                   tt.dot(breakpointsIndMat, shifts)[xIndex, flatPopIndex] +
                   tt.dot(periodic_bases, periodics)[xIndex, flatPopIndex])
        
        sigma_y = pm.HalfCauchy('sigma_y', 5., shape=N)
        nu_y = pm.HalfCauchy('nu_y', 5.)
        y = pm.StudentT('y',
                        nu=clip0to(nu_y),
                        mu=linear,
                        sd=clip0to(sigma_y[flatPopIndex]),
                        observed=flatProportion)
        
    
    return basic_model

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    with open('jobArraySelectedConditionsSoftBound/' + 
              CONDITION_FILE, 
              'rb') as f:
        condIn = pickle.load(f)
        rawSample = condIn['Data']
        condName = condIn['Condition']
    
    (flatX, flatPopIndex, flatPopSize, 
     flatObservation, flatAgeIndex, outKeys) = sample2Inputs(rawSample)
    
    breakpoints = 1 + np.array([52, 104, 156, 208, 261, 313, 365, 417, 469, 522])
    
    model = mcmc_Model(N=max(flatPopIndex)+1, 
                       breakpoints=breakpoints, 
                       ageBarsNum=AGE_GROUP_NUM,
                       flatAgeIndex=flatAgeIndex, 
                       flatX=flatX, 
                       flatPopIndex=flatPopIndex, 
                       flatPopSize=flatPopSize, 
                       flatObservation=flatObservation)
    
    for RV in model.basic_RVs:
        print(RV.name, RV.logp(model.test_point))

    with model:
        start, step = pm.init_nuts(init='advi', chains=NUM_CHAINS, n_init=200000)
        trace = pm.sample(draws=0,
                          chains=NUM_CHAINS,
                          tune=500,
                          start=start,
                          step=step,
                          discard_tuned_samples=False,
                          obj_optimizer=pm.adam())
    
    with open('tuned500/tunedTrace_{}.bpkl3'.format(COND_NUM), 'wb') as buff:
        pickle.dump({'condition': condName,
                     'model': model, 
                     'trace': trace,
                     'step': step}, buff)