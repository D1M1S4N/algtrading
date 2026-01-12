import torch
from torch import nn,Tensor
from torch.nn import Module
import math

class PositionalEncoder(nn.Module):
    def __init__(
        self,
        dropout: float=0.1,
        max_seq_len: int=5000,
        d_model: int=512,
        batch_first: bool=False
        ):

        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)

        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(self.x_dim)]

        return self.dropout(x)

class MHA(Module):
  def __init__(self, kv_dim, q_dim, n_heads=1, attn_pdrop=0., resid_pdrop=0.):
        super().__init__()
        self.n_embd = q_dim
        self.n_heads = n_heads
        assert self.n_embd % self.n_heads == 0
        # key, query, value projections
        self.key = torch.nn.Linear(kv_dim, self.n_embd)
        self.query = torch.nn.Linear(q_dim, self.n_embd)
        self.value = torch.nn.Linear(kv_dim, self.n_embd)
        # regularization
        self.attn_drop = torch.nn.Dropout(attn_pdrop)
        self.resid_drop = torch.nn.Dropout(resid_pdrop)
        # output projection
        self.proj = torch.nn.Linear(self.n_embd, q_dim)

        self.register_buffer("bias", torch.tril(torch.ones(q_dim, q_dim))
                                     .view(1, 1, q_dim, q_dim))

  def forward(self, kv, q, mask = None):
      B, M, C = kv.size()
      B, N, D = q.size()
      # calculate query, key, values for all heads in batch and move head forward to be the batch dim
      k = self.key(kv).view(B, M, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, M, hs)
      q = self.query(q).view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, N, hs)
      v = self.value(kv).view(B, M, self.n_heads, D // self.n_heads).transpose(1, 2) # (B, nh, M, hs)
      # attention (B, nh, N, hs) x (B, nh, hs, M) -> (B, nh, N, M)
      att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
      #att = att.masked_fill(self.bias[:,:,:N,:M] == 0, float('-inf'))
      att = torch.nn.functional.softmax(att, dim=-1)
      att = self.attn_drop(att)
      y = att @ v # (B, nh, N, M) x (B, nh, M, hs) -> (B, nh, N, hs)
      y = y.transpose(1, 2).contiguous().view(B, N, D) # re-assemble all head outputs side by side
      return self.resid_drop(self.proj(y)) # B, N, D

class Block(Module):
  def __init__(self, kv_dim, q_dim, n_heads=1, attn_pdrop=0., resid_pdrop=0.,tgt_mask=None,memory_mask=None):
        super().__init__()
        self.ln1_kv = torch.nn.LayerNorm(kv_dim)
        self.ln1_q = torch.nn.LayerNorm(q_dim)
        self.ln2 = torch.nn.LayerNorm(q_dim)
        self.attn = MHA(kv_dim, q_dim, n_heads, attn_pdrop, resid_pdrop)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(q_dim, 4 * q_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * q_dim, q_dim),
            torch.nn.Dropout(resid_pdrop),
        )

  def forward(self, kv, q, mask=None):
      x = q + self.attn(self.ln1_kv(kv), self.ln1_q(q), mask)
      x = x + self.mlp(self.ln2(x))
      return x

class TransformerModel(Module):
  def __init__(self,
    input_size: int,
    dec_seq_len: int,
    batch_first: bool,
    dim_val: int=512,
    n_encoder_layers: int=4,
    n_heads: int=8,
    dropout_pos_enc: float=0.1,
    dropout_enc:float=0.1,
    out_predicted_features: int=1,
    max_seq_len: int=1000,
    option: float='pre-enc',
    seq_len: int=30
    ):

    super().__init__()

    self.dec_seq_len=dec_seq_len
    self.seq_len=seq_len
    self.option=option

    self.encoder_input_layer = nn.Linear(
      in_features=input_size,
      out_features=dim_val
    )

    self.positional_encoding_layer=PositionalEncoder(
        d_model=dim_val,
        dropout=dropout_pos_enc,
        max_seq_len=max_seq_len,
    )

    encoder_layer=nn.TransformerEncoderLayer(
        d_model=dim_val,
        nhead=n_heads,
        batch_first=batch_first,
        dropout=dropout_enc
    )

    self.encoder=nn.TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=n_encoder_layers,
        norm=None
    )

    self.linear_mapping=nn.Linear(
        in_features=dim_val*seq_len,
        out_features=out_predicted_features
    )

    if option=='pre-enc':
      self.pre_enc=nn.TransformerEncoderLayer(1,nhead=1,batch_first=batch_first)

    self._init_weights()

  def _init_weights(module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)

  def configure_optimizers(self, learning_rate,betas,weight_decay):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        #assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    #% (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

  def forward(self,src: Tensor,src_mask,tgt_mask) -> Tensor:

    if self.option=='pre-enc':
      src_=src.unsqueeze(3)
      src=[]
      for i in range(len(src_)):
        src.append(self.pre_enc(src_[i,...]))

      src=torch.stack(src)
      src=src.squeeze(3)

    src=src.type('torch.FloatTensor')

    src=self.encoder_input_layer(src)

    src=self.positional_encoding_layer(src)

    src=self.encoder(src=src)
    
    decoder_output=self.linear_mapping(src.flatten(1,2))

    return decoder_output
  
if __name__=='__main__':
  model=TransformerModel(input_size=1,dec_seq_len=1,batch_first=True,dim_val=512,n_encoder_layers=4,n_heads=8,dropout_pos_enc=0.1,dropout_enc=0.1,out_predicted_features=1,max_seq_len=1000,option='no pre-enc',seq_len=100)
  print(model)
  x=torch.randn(64,100,1)
  print(x.shape)
  print(model(x,None,None).shape)