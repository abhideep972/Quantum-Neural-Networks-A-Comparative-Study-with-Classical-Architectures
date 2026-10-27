from math import pi
import numpy as np
from scipy.special import logsumexp

class EffectiveDimension:
    def __init__(self, model, num_thetas, num_inputs, seed=0):
        self.model=model; self.d=model.d; self.num_thetas=num_thetas; self.num_inputs=num_inputs
        rng=np.random.default_rng(seed)
        params=rng.uniform(model.thetamin,model.thetamax,size=(num_thetas,model.d))
        self.params=np.repeat(params,num_inputs,axis=0)
        x=rng.normal(0,1,size=(num_inputs,model.inputsize))
        self.x=np.tile(x,(num_thetas,1))

    def get_fhat(self):
        grads=self.model.get_gradient(params=self.params,x=self.x)
        output=self.model.forward(params=self.params,x=self.x)
        fishers=self.model.get_fisher(gradients=grads,model_output=output)
        fisher_trace=float(np.trace(np.average(fishers,axis=0)))
        if not np.isfinite(fisher_trace) or fisher_trace <= 0:
            raise ValueError(f'Invalid Fisher trace: {fisher_trace}')
        fisher=np.average(np.reshape(fishers,(self.num_thetas,self.num_inputs,self.d,self.d)),axis=1)
        return self.d*fisher/fisher_trace, fisher_trace

    def eff_dim(self,f_hat,n):
        result=[]
        for ns in n:
            if ns <= 1: raise ValueError('n must be > 1')
            Fhat=f_hat*ns/(2*pi*np.log(ns))
            logdet=np.linalg.slogdet(np.eye(self.d)+Fhat)[1]
            r=logdet/2
            denom=np.log(ns/(2*pi*np.log(ns)))
            result.append(2*(logsumexp(r)-np.log(self.num_thetas))/denom)
        return result
