import torch.nn as nn
import torch as t
import torch.nn.functional as F

class Lt_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u: t.Tensor,
                embedings: t.Tensor):
        u = u.reshape(u.shape[0], 1, u.shape[1])
        embedings = t.transpose(embedings, 2, 1)
        assert len(u.shape) == 3
        assert len(embedings.shape) == 3
        #print(u.shape)
        #print(embedings.shape)
        #assert u.shape[0] == embedings.shape[0]
        #assert u.shape[2] == embedings.shape[2]
        
        # embedding =[48,100,1280]
        # u = [48,1,1280]
        
        # embeddings是wav2vec2输出的c
        

        embeding1 = embedings[:, :, :embedings.shape[2] - 1]
        # print(f'22222222,{embeding1.shape}')    #（128,768,48）取前48帧，最后一帧不要

        embedings2 = embedings[:, :, 1:]
        # print(f'3333333333,{embedings2.shape}')     #（128,768,48）取后48帧，第一帧不要

        embedings = embeding1 - embedings2  # 直接减
        # print(f'444444444,{embedings.shape}')
        embedings = t.transpose(embedings, 1, 2)
        # print(f'55555555555555,{embedings.shape}')

        # out = t.matmul(u,embedings)
        out = F.cosine_similarity(u, embedings, dim=-1)
        return t.sum(F.relu(out))