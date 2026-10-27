from pathlib import Path
import argparse, numpy as np, torch
from torch import nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
class ANN(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(4,1,bias=False),nn.LeakyReLU(),nn.Linear(1,1,bias=False),nn.LeakyReLU(),nn.Linear(1,1,bias=False),nn.LeakyReLU(),nn.Linear(1,2,bias=False)).double()
 def forward(self,x): return self.net(x)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--runs',type=int,default=100); p.add_argument('--epochs',type=int,default=100); a=p.parse_args(); X=normalize(load_iris().data[:100]); y=load_iris().target[:100]; X,y=shuffle(X,y,random_state=0); X=torch.tensor(X,dtype=torch.float64); y=torch.tensor(y,dtype=torch.long); out=ROOT/'results'/'trainability'/'classical'; out.mkdir(parents=True,exist_ok=True)
 for r in range(a.runs):
  torch.manual_seed(r); m=ANN(); opt=torch.optim.Adam(m.parameters(),lr=.1); lossfn=nn.CrossEntropyLoss(); losses=[]
  for _ in range(a.epochs): opt.zero_grad(); loss=lossfn(m(X),y); losses.append(float(loss)); loss.backward(); opt.step()
  np.save(out/f'classical_loss_{r}.npy',losses)
if __name__=='__main__': main()
