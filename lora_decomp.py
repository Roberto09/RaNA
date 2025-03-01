import torch

class LoRaDecomp():
    def __init__(self, module, Q, B, X, out=None):
        self.module = module
        self.Q, self.B, self.X = Q, B, X
        if out is None:
            with torch.no_grad():
                self.out = (module.weight.detach() @ X.cuda()).T.cpu()
        else: self.out = out
    
    def get_errs(self):
        with torch.no_grad():
            B,X = self.B, self.X
            if self.B.dtype == torch.float16:
                B,X = B.to(torch.float32), X.to(torch.float32)
            return (B.T.cuda() @ X.cuda()).cpu()