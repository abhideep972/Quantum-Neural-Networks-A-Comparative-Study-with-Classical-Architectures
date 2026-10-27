from pathlib import Path
import sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from qiskit.circuit.library import z_feature_map, real_amplitudes
from qnn_repro import QuantumNeuralNetwork, EffectiveDimension

def test_qnn_forward_gradient_fisher():
 q=QuantumNeuralNetwork(z_feature_map(2,reps=1),real_amplitudes(2,reps=1)); x=np.array([[0.1,0.2],[0.2,0.3]]); p=np.full((2,q.d),0.2); y=q.forward(p,x); assert y.shape==(2,2); assert np.allclose(y.sum(1),1,atol=1e-10); g=q.get_gradient(p,x); f=q.get_fisher(g,y); assert f.shape==(2,q.d,q.d); assert np.isfinite(f).all()
def test_small_effective_dimension():
 q=QuantumNeuralNetwork(z_feature_map(2,reps=1),real_amplitudes(2,reps=1)); ed=EffectiveDimension(q,2,2); f,tr=ed.get_fhat(); vals=ed.eff_dim(f,[1000,2000]); assert f.shape==(2,q.d,q.d); assert np.isfinite(vals).all(); assert tr>0
