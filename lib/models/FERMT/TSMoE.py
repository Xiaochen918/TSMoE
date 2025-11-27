import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class Attention_cross(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TS_Decoder(nn.Module):

    def __init__(self, dim, num_heads=12, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention_cross(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_fore, x_back):

        x = torch.cat([x_fore, x_back], dim=1)
        x = x + self.drop_path(self.attn(self.norm(x)))

        return x



class MambaBlock(nn.Module):
    def __init__(self,dt_scale, d_model,d_inner,dt_rank, d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor):
        super().__init__()
        #  projects block input from D to 2*ED (two branches)
        self.dt_scale = dt_scale
        self.d_model = d_model
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)

        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner,
                                kernel_size=d_conv, bias=conv_bias,
                                groups=self.d_inner,
                                padding=(d_conv - 1)//2)

        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(self.d_inner))

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x, h=None):
        #  x : (B,L, D)
        # h : (B,L, ED, N)
        #  y : (B, L, D)
        xz = self.in_proj(x)  # (B, L,2*ED)
        x, z = xz.chunk(2, dim=-1)  #  (B,L, ED), (B,L, ED)
        x_cache = x.permute(0,2,1)#(B, ED,L)

        #  x branch
        x = self.conv1d(x_cache).permute(0,2,1) #  (B,L , ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)
        #y->B,L,ED;h->B,L,ED,N

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, L, D)

        return output, h

    def ssm_step(self, x, h=None):
        #  x : (B, L, ED)
        #  h : (B, L, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.dt_rank, self.d_state, self.d_state],
                                  dim=-1)  #  (B, L,dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B,L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B,L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L,ED, N)

        if h is None:
            h = torch.zeros(x.size(0), x.size(1), self.d_inner, self.d_state, device=deltaA.device)  #  (B, L, ED, N)

        h = deltaA * h + BX  #  (B, L, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x#B,L,ED

        #  todo : pq h.squeeze(1) ??
        return y, h


class Mamba_Neck(nn.Module):
    def __init__(self, in_channel=512,d_model=512,d_inner=1024,bias=False,n_layers=6,dt_rank=32,d_state=16,d_conv=3,dt_min=0.001,
                 dt_max=0.1,dt_init='random',dt_scale=1.0,conv_bias=True,dt_init_floor=0.0001):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.bias = bias
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.dt_scale = dt_scale
        self.num_channels = self.d_model
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [ResidualBlock(dt_scale,d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor)
             for _ in range(n_layers)])
        # self.norm_f = RMSNorm(config.d_model)

    def forward(self, x, h=None):
        #  x : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        #  y : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)
        for i in range(self.n_layers):
            x, h = self.layers[i](x, h)
            # print(i)

        return x, h

class ResidualBlock(nn.Module):
    def __init__(self,dt_scale, d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor):
        super().__init__()

        self.mixer = MambaBlock(dt_scale,d_model,d_inner,dt_rank,d_state,bias,d_conv,conv_bias,dt_init,dt_max,dt_min,dt_init_floor)
        self.norm = RMSNorm(d_model)

    def forward(self, x, h):
        #  x : (B, L, D)
        # h : (B, L, ED, N)
        #  output : (B,L, D)

        x = self.norm(x)
        output, h = self.mixer(x, h)
        output = output + x
        return output, h


#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output



class SelectToken(nn.Module):
    def __init__(self, dim, topk_fore=8, topk_back=4, winSize=2):
        super().__init__()

        self.dim = dim
        self.topk_fore = topk_fore
        self.topk_back = topk_back
        self.winSize = winSize

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))

        # Temporal State MoE (TS-MoE)
        self.dim_m = dim
        self.n_layers = 1
        self.d_state = 16
        self.dt_rank = self.dim_m//16
        self.foreground_SSM = Mamba_Neck(in_channel=self.dim_m,d_model=self.dim_m,d_inner=2*self.dim_m,
                                      n_layers=self.n_layers,dt_rank=self.dt_rank,d_state=self.d_state)
        self.background_SSM = Mamba_Neck(in_channel=self.dim_m,d_model=self.dim_m,d_inner=2*self.dim_m,
                                      n_layers=self.n_layers,dt_rank=self.dt_rank,d_state=self.d_state)
        
        self.TS_Decoder = TS_Decoder(dim=self.dim)

    def forward(self, z, x):
        # template
        B, N_t, C = z.shape
        h_t = int(math.sqrt(N_t))
        z = z.permute(0,2,1).reshape(B,C,h_t,h_t)
        z_max = (self.maxpool(z)).permute(0,2,3,1).reshape(B,1,C)

        # search region
        N_s = x.shape[1]
        h_s = int(math.sqrt(N_s))
        win_Size_all = int(self.winSize*self.winSize)
        win_Num_H = h_s//self.winSize

        # Appearance Branch
        sim_x = ((z_max @ x.transpose(-2,-1))/C)
        index_z = torch.topk(sim_x,k=self.topk_fore,dim=-1)[1]
        index_z = index_z.permute(0,2,1).expand(-1,-1,C)
        x_fore = torch.gather(x,dim=1,index=index_z)
        x_fore = self.foreground_SSM(x_fore)[0]

        # Environment Branch
        sim_x = sim_x.reshape(B,win_Num_H,self.winSize,win_Num_H,self.winSize).permute(0,1,3,2,4)
        sim_x = (sim_x.reshape(B,-1,win_Size_all)).mean(dim=-1)
        index_x_T = torch.topk(sim_x,k=self.topk_back,dim=-1)[1] # [B,win_topk]
        index_x_T = index_x_T.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1,-1,win_Size_all,C)
        x_back = x.reshape(B,win_Num_H,self.winSize,win_Num_H,self.winSize,C)
        x_back = x_back.permute(0,1,3,2,4,5).reshape(B,-1,win_Size_all,C)
        x_back = torch.gather(x_back,dim=1,index=index_x_T).reshape(B,-1,C)
        x_back = self.background_SSM(x_back)[0]
        x_back = x_back.reshape(B,self.topk_back,win_Size_all,C).mean(dim=-2)

        # Appearance-Environment Fusion
        x_ext = self.TS_Decoder(x_fore, x_back)

        return x_ext