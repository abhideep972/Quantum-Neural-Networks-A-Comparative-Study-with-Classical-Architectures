from pathlib import Path
import argparse, sys, numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).parent))
from qiskit import QuantumCircuit
from qiskit.circuit.library import z_feature_map, real_amplitudes
from qnn_repro import ADAM
from common import iris_binary, parity_prob, cross_entropy

def main():
 p=argparse.ArgumentParser(); p.add_argument('--runs',type=int,default=100); p.add_argument('--maxiter',type=int,default=100); p.add_argument('--samples-per-class',type=int,default=49); a=p.parse_args()
 data=iris_binary(a.samples_per_class); fm=z_feature_map(feature_dimension=4,reps=1); vf=real_amplitudes(4,reps=1); qc=QuantumCircuit(4); qc.compose(fm,inplace=True); qc.compose(vf,inplace=True); fp=list(fm.parameters); vp=list(vf.parameters)
 samples=[(label,x) for label,xs in data.items() for x in xs]
 def cost(theta):
  total=0.0
  for label,x in samples:
   bind={**dict(zip(fp,x)),**dict(zip(vp,theta))}; total+=cross_entropy(parity_prob(qc,bind),label)
  return total/len(samples)
 out=ROOT/'results'/'trainability'/'easy_qnn'; out.mkdir(parents=True,exist_ok=True)
 for i in range(a.runs):
  init=2*np.pi*np.random.default_rng(i).random(len(vp)); opt=ADAM(maxiter=a.maxiter,lr=0.1); theta,val,_,loss=opt.optimize(len(vp),cost,initial_point=init); np.save(out/f'quantum_loss_easy_{i}.npy',loss); np.save(out/f'params_{i}.npy',theta); print(i,val)
if __name__=='__main__': main()
