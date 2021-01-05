import time
time_start=time.time()

import matplotlib.pyplot as plt
import numpy as np
import pints
import pints.plot
import pints.toy
import pandas as pd
import math
from scipy.stats import binom
import pints.toy as toy
from datetime import datetime


# Create a wrapper around the logistic model, turning it into a 1d model
class Model(pints.ForwardModel):
    def __init__(self):
        self.model = toy.LogisticModel()
    def simulate(self, x, times):
        return self.model.simulate([0.01, x[0]], times)
    def simulateS1(self, x, times):
        values, gradient = self.model.simulateS1([0.01, x[0]], times)
        gradient = gradient[:, 0]
        return values, gradient
    def n_parameters(self):
        return 1

def flatten(matrix):
    l = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            l.append(matrix[i][j])
    return l

def rank_statistic(thetalist,thetai):
    count = 0
    for elem in thetalist:
        if elem < thetai:
            count += 1
    return count

N = 1000
d = 0

model = Model()

# Create a uniform prior over carrying capacity
log_prior = pints.UniformLogPrior(
    [400],
    [600]
)

Ldash = 1000
L = 10
# the parameter we consider
i = 0
ax = plt.hist([], bins = range(L+2), align='left')
rankstats = []
columns = [str(x) for x in range(N)]
df = pd.DataFrame(columns=columns)
for n in range(N):
    print(n)
    theta = log_prior.sample(n=1)[0]
    times = np.linspace(1, 1000, 50)
    org_values = model.simulate(theta,times)
    # Add noise
    noise = 10
    ys = org_values + np.random.normal(0, noise, org_values.shape)
    # Create an object with links to the model and time series
    problem = pints.SingleOutputProblem(model, times, ys)
    # Create a log-likelihood function (adds an extra parameter!)
    log_likelihood = pints.GaussianKnownSigmaLogLikelihood(problem, [noise])
    
    log_prior_incorrect = pints.UniformLogPrior(
    [200],
    [800]
)
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)
    
    # Choose starting points for 3 mcmc chains
    xs = [
        theta,
        theta*1.01,
        theta*0.99
    ]
    isinf=False
    for x in xs:
        if (math.isinf(log_posterior.evaluateS1(x)[0])):
            isinf = True
            d+=1
            break
    if (isinf==True):
        continue
        
  
    # Create mcmc routine with three chains
    mcmc = pints.MCMCController(log_posterior, 3, xs, method=pints.HaarioACMC)
    

    # Add stopping criterion
    sample_size = Ldash
    mcmc.set_max_iterations(sample_size)

    # Start adapting after 1000 iterations
    mcmc.set_initial_phase_iterations(sample_size//4)

    # Disable logging mode
    mcmc.set_log_to_screen(False)

    # Run!
    #print('Running...')
    
    chains = mcmc.run()
    #choose the first chain
    chain0 = chains[0]
    
    N_effective1 = pints.effective_sample_size(chain0)
    N_effective1[i] = int(round(N_effective1[i]))

    if N_effective1[i] < L:
        mcmc = pints.MCMCController(log_posterior, 3, xs, method=pints.HaarioACMC)
        sample_size = Ldash*L//N_effective1[i]
        mcmc.set_max_iterations(sample_size)
        mcmc.set_initial_phase_iterations(sample_size//4)
        mcmc.set_log_to_screen(False)
        chains = mcmc.run()
    s = sample_size//4+1
    #HMC: s = 1
    b = False
    while s < sample_size-L:
        chains_cut = chains[:,sample_size//4:s+1]
        #HMC: chains_cut = chains[:,0:s+1]
        rhat = pints.rhat(chains_cut)
        s+=1
        if rhat[0] < 1.05:
            b = True
            break
        
    if b == False:
        d += 1
        continue
    chain0 = chains[0]
    
    N_effective2 = pints.effective_sample_size(chain0)
    chain = chain0[s:]
    
    
    N_effective3 = pints.effective_sample_size(chain)

    k = len(chain)//L
    
    res = chain[0::k][:L]

    N_effective4 = pints.effective_sample_size(res)
   
    resList = flatten((res[:, [i]]))
    rankstat = rank_statistic(res[:, [i]],theta[i])
    rankstats.append(rankstat)
    series = [theta[i],N_effective1[i],N_effective2[i],N_effective3[i],N_effective4[i]]+resList
   
    df[columns[n]] = pd.Series(series)
df.index=['sample from the prior', 'effective_sample_size1', 'effective_sample_size2', 'effective_sample_size3', 'effective_sample_size4']+['final_chain']*L


current_date_and_time = datetime.now()
current_date_and_time_string = str(current_date_and_time)
df.to_excel('./KnownGrowthRateLogisticsModel_HaarioACMC_convergence_record'+current_date_and_time_string+'.xlsx')


plt.hist(rankstats, bins = range(L+2),align='left')
plt.axhline(y=(N-d)/(L+1), color='r', linestyle='-')
plt.axhline(y=binom.ppf(0.005, N-d, 1/(L+1)), color='b')
plt.axhline(y=binom.ppf(0.995, N-d, 1/(L+1)), color='b')
plt.savefig('./KnownGrowthRateLogisticsModel_HaarioACMC_convergence_record'+current_date_and_time_string+'.png',dpi=500,bbox_inches = 'tight')
time_end=time.time()
print('total running time',time_end-time_start)
plt.show()
