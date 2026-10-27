import numpy as np

class ADAM:
    """Small self-contained Adam optimizer replacing removed qiskit.aqua ADAM."""
    def __init__(self,maxiter=100,tol=1e-6,lr=1e-3,beta_1=0.9,beta_2=0.99,eps=1e-6,noise_factor=1e-8):
        self.maxiter=maxiter; self.tol=tol; self.lr=lr; self.beta_1=beta_1; self.beta_2=beta_2
        self.eps=eps; self.noise_factor=noise_factor

    def _num_grad(self,fun,x):
        g=np.empty_like(x,dtype=float)
        for i in range(len(x)):
            xp=x.copy(); xm=x.copy(); xp[i]+=self.eps; xm[i]-=self.eps
            g[i]=(fun(xp)-fun(xm))/(2*self.eps)
        return g

    def optimize(self,num_vars,objective_function,gradient_function=None,variable_bounds=None,initial_point=None):
        if initial_point is None: initial_point=np.random.default_rng(0).random(num_vars)
        x=np.asarray(initial_point,dtype=float).copy(); m=np.zeros_like(x); v=np.zeros_like(x); losses=[]
        gradfun=gradient_function or (lambda z:self._num_grad(objective_function,z))
        for t in range(1,self.maxiter+1):
            g=np.asarray(gradfun(x),dtype=float)
            m=self.beta_1*m+(1-self.beta_1)*g; v=self.beta_2*v+(1-self.beta_2)*g*g
            lr_eff=self.lr*np.sqrt(1-self.beta_2**t)/(1-self.beta_1**t)
            xnew=x-lr_eff*m/(np.sqrt(v)+self.noise_factor)
            losses.append(float(objective_function(xnew)))
            if np.linalg.norm(x-xnew)<self.tol: x=xnew; break
            x=xnew
        return x,float(objective_function(x)),len(losses),losses
