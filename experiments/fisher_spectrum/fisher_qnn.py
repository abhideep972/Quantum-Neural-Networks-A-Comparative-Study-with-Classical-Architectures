from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse, numpy as np
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qnn_repro import QuantumNeuralNetwork, EffectiveDimension

def main():
 p=argparse.ArgumentParser(); p.add_argument('--num-inputs',type=int,default=100); p.add_argument('--num-thetas',type=int,default=100); p.add_argument('--blocks',type=int,default=9); a=p.parse_args()
 fm=zz_feature_map(feature_dimension=4,reps=2); vf=real_amplitudes(4,reps=a.blocks)
 ed=EffectiveDimension(QuantumNeuralNetwork(fm,vf),a.num_thetas,a.num_inputs)
 f,_=ed.get_fhat(); out=ROOT/'results'/'fisher_spectrum'; out.mkdir(exist_ok=True)
 np.save(out/'4qubits_9layer_f_hats_dep2.npy',f)
 print('saved',out/'4qubits_9layer_f_hats_dep2.npy',f.shape)
if __name__=='__main__': main()
