import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple, deque
import copy

PATCH_ATTENTION_SIZE = 4096
DEFAULT_NUMBER_HUMANS = 1

class TransformerMemory(object):

    def __init__(self, num_environments, capacity):
      self.num_environments = num_environments
      self.capacity = capacity
      self.full_flag = False
      self.memory = deque([], maxlen=capacity)
      self.current_first_size = self.num_environments

    def push(self, emb_feature):
        """Save a transition"""
        new_first_size = emb_feature.data.shape[0]

        if new_first_size != self.current_first_size:
          self.memory.clear()
          self.current_first_size = new_first_size
          self.full_flag = False


        self.memory.append(emb_feature.unsqueeze(dim=1))
        if len(self.memory) == self.capacity:
          self.full_flag = True

    def sample(self):
      return self.memory
    

    def __len__(self):
        return len(self.memory)


class SelfPatchAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.patch_attention = nn.Transformer(d_model=256, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=128)
    self.output_size = 128
    self.reduction = nn.Sequential(
      nn.Linear(256, 64),
      nn.GELU(),
      nn.Linear(64, 16),
      nn.GELU(),
      nn.Flatten(),
      nn.Linear(PATCH_ATTENTION_SIZE, 1024),
      nn.GELU(),
      nn.Linear(1024, self.output_size),
      nn.GELU(),
    )
  
  def forward(self, x):
    x = torch.transpose(x, 1, 2)
    x = self.patch_attention(x, x)
    x = self.reduction(x)
    
    return x

class SeriesAttention(nn.Module):
  def __init__(self, transformer_memory_size, is_series_attention, hidden_size):
    super().__init__()
    self.transformer_memory_size = transformer_memory_size
    self.flatten_layer = nn.Flatten()
    self._series_attention = is_series_attention
    if is_series_attention:
      self.series_attention = nn.MultiheadAttention(hidden_size, 16)
    else:
      class CustomIndentity(nn.Module):
        def __init__(self):
          super().__init__()
          self.layer = nn.Identity()
        
        def forward(self, x, _):
          return self.layer(x)
      
      self.series_attention = CustomIndentity()

  def forward(self, x):
    output = []
    attention = []
    for _x in x:
      _x = _x.cuda().clone().detach()
      attn_output, attn_output_weights = self.series_attention(_x, _x, _x)
      attn_matrix = attn_output_weights
      y = attn_matrix.narrow(0, -1, 1).narrow(1, 0, attn_matrix.data.shape[1]-1)
      s = y.sum(-1)
      output.append(attn_output.unsqueeze(0))
      attention.append((y.T/s).T.unsqueeze(0))

    output = torch.cat(output).narrow(1, -DEFAULT_NUMBER_HUMANS, 1).squeeze(1)
    attention = torch.cat(attention)
    return output, attention

class CrowdDynamicNet(nn.Module):
  def __init__(self, num_environments):
    super().__init__()
    self.num_environments = num_environments
    self.layer = nn.GRU(input_size=512, hidden_size=3, num_layers=self.num_environments)
  
  def forward(self, x, hxs=None):
    if hxs is None:
      hxs = self._init_hxs(device=x.device)
    
    x, hxs = self.layer(x, hxs)

    return x, hxs

  def _init_hxs(self, device):
    return torch.zeros((self.num_environments, DEFAULT_NUMBER_HUMANS, 3)).to(device)
