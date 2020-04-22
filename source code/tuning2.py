import pickle
import pymc3 as pm
import sys

# Constants preparation -------------------------------------------------------

COND_NUM = sys.argv[1]
NUM_CHAINS = 1

with open('tuned500/tunedTrace_{}.bpkl3'.format(COND_NUM), 'rb') as buff:
    res = pickle.load(buff)
    condName = res['condition']
    model = res['model']
    trace = res['trace']
    step = res['step']
    
with model:
    traceContinued = pm.sample(draws=500,
                               chains=NUM_CHAINS,
                               tune=500,
                               trace=trace,
                               step=step,
                               discard_tuned_samples=False,
                               obj_optimizer=pm.adam())
    
with open('tuned1000drawn500/trace_{}.bpkl3'.format(COND_NUM), 'wb') as buff:
    pickle.dump({'condition': condName,
                 'model': model, 
                 'trace': trace,
                 'step': step}, buff)