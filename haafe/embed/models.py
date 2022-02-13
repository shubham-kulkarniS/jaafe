import os
from turtle import forward
from torch import nn
import pase
from pase.models.frontend import wf_builder



vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
vggish.cuda().eval()



# pnet = wf_builder(os.path.join(*pase.__path__,'cfg/frontend/PASE+.cfg'))
# pnet.load_pretrained(os.path.join(*pase.__path__,'cfg/FE_e199.ckpt'), load_last=True, verbose=True)


# pase = wf_builder('/kaggle/input/problem-agnostic-speech-encoder-pase/pase-master/pase-master/cfg/frontend/PASE+.cfg')
# pase.load_pretrained('/kaggle/input/problem-agnostic-speech-encoder-pase/FE_e199.ckpt', load_last=True, verbose=True)
    

class ModPASE(nn.Module):
    def __init__(self, net, input_size=256, hidden_size=64, num_layers=3, seqlen=800):
        super(ModPASE, self).__init__()       
        
        self.pase = net
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seqlen = seqlen
            
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)#, dropout=0.2, bidirectional=False)

        # Adaptive Average Pooling (output has always the same seqlen size)
        self.pool = nn.AdaptiveAvgPool1d(self.seqlen)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),           
            nn.Dropout(0.1),            
            nn.Linear(hidden_size, 1),
        )           
          
        
    def forward(self, x):
        
        # PASE needs an additional dimension
        x.unsqueeze_(1)
        # (B,1,16000 x max_seconds)
        
        out = self.pase(x)
        # out shape: (B,256,100 x max_seconds)

        out = self.pool(out)
        # out shape: (B,256,seqlen)

        out = out.transpose(1,2)
        # out shape: (B,seqlen,256)

        out, _ = self.lstm(out)
        # out shape: (B,seqlen,H)

        out, _ = out.max(dim=1)
        # out = out.mean(dim=1)
        # out shape: (B,H)

        out = self.classifier(out)
        # out shape: (B,1)

        return out.squeeze(-1)


class TwoLayer(nn.Module):
    def __init__(self, input_size=128, hidden_size=64):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),           
            nn.Dropout(0.1),            
            nn.Linear(hidden_size, 1),
        )           

    def forward(self, x):
        # x shape: (B,E)

        # Mutilayer perceptron
        out = self.classifier(x)
        # out shape: (B,1)

        # Remove last dimension
        return out.squeeze(-1)
        # return shape: (B)


class ModVGGish(nn.Module):
    def __init__(self,vggish,input_size,hidden_size) -> None:
        super().__init__()
        self.net = vggish
        self.classifier = TwoLayer(input_size=input_size,hidden_size=hidden_size)

    def forward(self,x):
        feat = self.net(x)
        return self.classifier(feat)
