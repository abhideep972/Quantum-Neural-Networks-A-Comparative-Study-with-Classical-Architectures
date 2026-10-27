import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .model_base import Model

class QuantumNeuralNetwork(Model):
    """Statevector QNN compatible with modern Qiskit (2.x).

    The legacy multiprocessing implementation was deliberately replaced with a
    deterministic serial batch loop so it also works on native Windows.
    """
    def __init__(self, feature_map, var_form, post_processing=None):
        super().__init__()
        if feature_map.num_qubits != var_form.num_qubits:
            raise ValueError('feature map and variational form must use the same qubit count')
        self.feature_map=feature_map; self.var_form=var_form
        self.feature_parameters=list(feature_map.parameters)
        self.var_parameters=list(var_form.parameters)
        self.d=len(self.var_parameters)
        self.inputsize=len(self.feature_parameters)
        self.circuit=QuantumCircuit(feature_map.num_qubits)
        self.circuit.compose(feature_map,inplace=True)
        self.circuit.compose(var_form,inplace=True)
        self.post_processing=post_processing or self._parity()
        self.outputsize=len(self.post_processing)

    def _get_params_dict(self,params,x):
        if len(x)!=len(self.feature_parameters): raise ValueError('input dimension does not match feature map')
        if len(params)!=len(self.var_parameters): raise ValueError('parameter dimension does not match variational form')
        return {**dict(zip(self.feature_parameters,x)), **dict(zip(self.var_parameters,params))}

    @staticmethod
    def _as_batches(params,x):
        p=np.asarray(params,dtype=float); d=np.asarray(x,dtype=float)
        if p.ndim==1: p=p[None,:]
        if d.ndim==1: d=d[None,:]
        if len(p)==1 and len(d)>1: p=np.repeat(p,len(d),axis=0)
        if len(d)==1 and len(p)>1: d=np.repeat(d,len(p),axis=0)
        if len(p)!=len(d): raise ValueError('params and x must have equal batch sizes or one batch size must be 1')
        return p,d

    def _get_probabilities(self,params,x):
        params,x=self._as_batches(params,x)
        probs=np.empty((len(x),2**self.circuit.num_qubits),dtype=float)
        for i,(theta,data) in enumerate(zip(params,x)):
            bound=self.circuit.assign_parameters(self._get_params_dict(theta,data),inplace=False)
            probs[i]=Statevector.from_instruction(bound).probabilities()
        return probs

    def forward(self,params,x):
        probs=self._get_probabilities(params,x)
        out=np.zeros((len(probs),self.outputsize),dtype=float)
        for j,indices in enumerate(self.post_processing.values()):
            out[:,j]=probs[:,indices].sum(axis=1)
        return out

    def get_gradient(self,params,x):
        params,x=self._as_batches(params,x)
        grads=np.empty((len(x),self.d,2**self.circuit.num_qubits),dtype=float)
        for j in range(self.d):
            plus=params.copy(); minus=params.copy()
            plus[:,j]+=np.pi/2; minus[:,j]-=np.pi/2
            grads[:,j,:]=0.5*(self._get_probabilities(plus,x)-self._get_probabilities(minus,x))
        return grads

    def get_fisher(self,gradients,model_output):
        gradients=np.asarray(gradients,dtype=float); output=np.asarray(model_output,dtype=float)
        fishers=np.zeros((len(gradients),self.d,self.d),dtype=float)
        eps=np.finfo(float).eps
        for k in range(len(gradients)):
            for label_idx,indices in enumerate(self.post_processing.values()):
                p=max(float(output[k,label_idx]),eps)
                dp=gradients[k,:,indices].sum(axis=1)
                score=dp/np.sqrt(p)
                fishers[k]+=np.outer(score,score)
        return fishers

    def _parity(self):
        even=[]; odd=[]
        for idx,bits in enumerate(itertools.product([0,1],repeat=self.circuit.num_qubits)):
            (even if sum(bits)%2==0 else odd).append(idx)
        return {'y1':even,'y2':odd}
