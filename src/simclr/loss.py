import torch
import torch.nn as nn

class NTXentLoss(nn.Module):
    def __init__(self, temp):
        super(NTXentLoss, self).__init__()
        self.temp = temp
    
    def forward(self, zi, zj):
        batch_size = zi.shape[0]
        z_proj = torch.cat((zi, zj), dim=0)
        cos_sim = torch.nn.CosineSimilarity(dim=-1)
        sim_mat = cos_sim(z_proj.unsqueeze(1), z_proj.unsqueeze(0))
        sim_mat_scaled = torch.exp(sim_mat/self.temp)
        r_diag = torch.diag(sim_mat_scaled, batch_size)
        l_diag = torch.diag(sim_mat_scaled, -batch_size)
        pos = torch.cat([r_diag, l_diag])
        diag_mat = torch.exp(torch.ones(batch_size * 2)/self.temp).cuda()
        logit = -torch.log(pos/(sim_mat_scaled.sum(1) - diag_mat))
        loss = logit.mean()
        return loss