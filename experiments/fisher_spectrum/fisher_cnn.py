from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse, numpy as np
from qnn_repro import ClassicalNeuralNetwork, EffectiveDimension

def main():
 p=argparse.ArgumentParser(); p.add_argument('--num-inputs',type=int,default=100); p.add_argument('--num-thetas',type=int,default=100); a=p.parse_args()
 ed=EffectiveDimension(ClassicalNeuralNetwork([4,4,4,2]),a.num_thetas,a.num_inputs); f,_=ed.get_fhat()
 out=ROOT/'results'/'fisher_spectrum'; out.mkdir(exist_ok=True); np.save(out/'fhat4_[4 4 4 2]_fisher.npy',f); print('saved',f.shape)
if __name__=='__main__': main()
