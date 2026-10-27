from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn import datasets, preprocessing
from qiskit.quantum_info import Statevector

def iris_binary(samples_per_class=49):
 x=preprocessing.normalize(datasets.load_iris().data); return {'A':x[:samples_per_class], 'B':x[50:50+samples_per_class]}
def parity_prob(circuit,bind):
 p=Statevector.from_instruction(circuit.assign_parameters(bind,inplace=False)).probabilities(); idx=np.arange(len(p)); parity=np.array([int(i).bit_count()%2 for i in idx]); return float(p[parity==0].sum())
def cross_entropy(pa,label):
 pa=np.clip(pa,1e-12,1-1e-12); return -np.log(pa if label=='A' else 1-pa)
