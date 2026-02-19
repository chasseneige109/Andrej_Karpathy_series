# nanochat / gpt.py

from functools import partial # 함수 인자 일부만 넣어서 함수처럼 쓰는 기능
from dataclasses import dataclass # Config Class 만들때사용

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0 # 분산학습정보 , GPU 여러대 사용시 1개만 출력되게
from nanochat.optim import MuonAdamW, DistMuonAdamW # Muon optimizer를 사용함.

from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig():
  sequence_len: int = 2048
  vocab_size: int = 32768 # 단어 사전의 단어 수
  n_layer: int = 12
  n_head: int = 6
  n_kv_head: int = 6 #(GQA)
  n_embd: int = 768

  window_pattern: str = "SSSL"

# Purely functional rmsnorm with no learnable params
def norm(x):
  return F.rms_norm(x, (x.size(-1),))

# Check whether ve or not ve
def has_ve(layer_idx, n_layer):
  return layer_idx % 2 == (n_layer - 1) % 2

# Positional embedding : RoPE
def apply_rotary_emb(x, cos, sin):
  assert x.ndim == 4
  d = x.shape[3] // 2
  x1, x2 = x[..., :d], x[..., d:]
  y1 = cos * x1 + sin * x2
  y2 = -sin* x1 + cos * x2
  return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
  def __init__(self, config, layer_idx):
    super().__init__()

    self.layer_idx = layer_idx
    self.n_head = config.n_head # Number of Head
    self.n_kv_head = config.n_kv_head # GQA라서 q와 k,v의 head 수가 다름
    self.n_embd = config.n_embd # total embedding Dimensions
    self.head_dim = self.n_embd // self.n_head # embedding Dimensions for one head
    assert self.n_embd % self.n_head == 0
    assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
    self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias = False)
    self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias = False)
    self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias = False)
    self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias = False)
    self.ve_gate_channels = 32 # n_embd 차원의 채널 중에서 몇 채널씩 묶어서 그룹을 만들지
    self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias = False) if has_ve(layer_idx, config.n_layer) else None

  def forward(self, x, ve, cos_sin, window_size, kv_cache):
    B, T, C = x.size()

    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    if ve is not None:
      ve = ve.view(B, T, self.n_kv_head, self.head_dim)
      gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
      v = v + gate.unsqueeze(-1) * ve #unsqueeze: 마지막에 차원 하나 더 늘리기

    cos, sin = cos_sin
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)

    # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context

    # Training Mode
    if kv_cache is None:
      y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    # Inference Mode
    else:
      k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
      # Implement both attention & kv_cache at Once
      y = flash_attn.flash_attn_with_kvcache(
      q, k_cache, v_cache,
      k=k, v=v,
      cache_seqlens=kv_cache.cache_seqlens,
      causal=True,
      window_size=window_size
      )

      if self.layer_idx == kv_cache.n_layers - 1:
        kv_cache.advance(T)

    # contiguous는 메모리 여기저기 흩어진 데이터를 찾아서 새로운 메모리 공간에 넣음.
    y = y.contiguous().view(B, T, -1)
    y = self.c_proj(y)
    return y


class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)


  def forward(self, x):
    x = self.c_fc(x)
    x = F.relu(x).square() # ReGLU , ReLU^2 기법. Swiglu도 좋다네요.
    x = self.c_proj(x)
    return x

class Block(nn.Module):
  def __init__(self, config, layer_idx):
    super().__init__()
    self.attn = CausalSelfAttention(config, layer_idx)
    self.mlp = MLP(config)

  def forward(self, x, ve, cos_sin, window_size, kv_cache):
    x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
    x = x + self.mlp(norm(x))
    return x



"""
Meta Device임. 파라미터가 수십억개인 모델은 init만 해도 메모리가 꽉차서 터져버리니까
모델 크기와 데이터 타입 정보만 있는 깡통 모델로 만듦. 그리고 추후에 init_weight, to_empty
같은 걸로 메모리를 할당해줌.
"""
# init에 값이 없어서, 실수하기 쉽다!
# has_ve처럼 파라미터를 확인하는 건 동작가능, but 실제값 확인하는 연산은 불가능
class GPT(nn.Module):
  def __init__(self, config, pad_vocab_size_to=64):
    super().__init__()
    self.config = config

    # Sliding window attn: SSSL
    self.window_sizes = self._compute_window_sizes(config)
    
    # Padded_vocab_size_to의 배수(32, 64 등)로 맞추기 : GPU 효율성을 위해.
    padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
    if padded_vocab_size != config.vocab_size:
      print0(f"padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
    
    # Transformer 뼈대: embedding + Block들
    self.transformer = nn.ModuleDict({
        "wte" : nn.Embedding(padded_vocab_size, config.n_embd,),
        "h" : nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
    })

    self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
    
    # Residual Connection시 , 모든 레이어에서 전레이어(x) * lamb1 + 입력층(x0) * lamb0 를 수행함.
    # 모든 레이어의 resid parameter들 만들기
    self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
    self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

    head_dim = config.n_embd // config.n_head
    kv_dim = config.n_kv_head * head_dim

    # value embedding
    self.value_embeds = nn.ModuleDict({str(i) : nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
    
    # RoPE - positional embedding
    # 학습시엔 알아서 끊겨서 학습하지만, 
    # 추론시엔 seq_len보다 길어질 수도 있으니 안전빵으로 10배 잡아놓기
    self.rotary_seq_len = config.sequence_len * 10 
    cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
    self.register_buffer("cos", cos, persistent=False) # 1. 장치 자동이동 2. 모델의 일부로 등록. self.cos가능
    self.register_buffer("sin", sin, persistent=False)


  # Ctrl-C, Ctrl-Ved
  @torch.no_grad() 
  def init_weights(self):
    """
        # Meta_device의 가중치 채워넣는 부분을 이 함수 1개에 몰빵

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """
    # Embedding and unembedding
    torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
    torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

    # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
    n_embd = self.config.n_embd
    s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
    for block in self.transformer.h:
        torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
        torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
        torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
        torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
        torch.nn.init.zeros_(block.mlp.c_proj.weight)

    # Per-layer scalars
    self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
    self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

    # Value embeddings (init like c_v: uniform with same std)
    for ve in self.value_embeds.values():
        torch.nn.init.uniform_(ve.weight, -s, s)

    # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
    for block in self.transformer.h:
        if block.attn.ve_gate is not None:
            torch.nn.init.zeros_(block.attn.ve_gate.weight)

    # Rotary embeddings
    head_dim = self.config.n_embd // self.config.n_head
    cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
    self.cos, self.sin = cos, sin

    # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
    if self.transformer.wte.weight.device.type == "cuda":
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)


  # RoPE
  """
  Q,K,V를 만들고 n_head 를 멀티헤드로 쪼갠 후 Q와 K의 head_dim 차원에 RoPE 수행
  이때 1,2 / 3,4 ... 이 아니라 1,65 / 2,66 등으로 묶어서 '단어 자신의 순서 Number' 에다가 cos, sin 행렬 곱함.
  cos, sin 행렬도 아래 함수에서 미리 생성해놓기.
  """
  def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
    
    # device가 지정되지 않았다면, 모델의 가장 첫 레이어인 wte의 위치를 기본장치로 설정.
    if device is None:
      device = self.transformer.wte.weight.device # wte가 nn.ModuleDict에 저장된 nn.Embedding Layer여서 weight 호출 가능.
    
    # Channel 주파스 설정. 쌍으로 묶을 거라 짝수로 뽑음
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) # arange()는 Tensor / range()는 List
    
    # inv_freq: 각 차원 쌍의 회전 속도 / 앞은 빠르게 뒤는 천천히
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    
    # 문장 속 단어들의 '위치 번호' 만들기
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq) # 벡터를 받아서 flatten 후 2차원 행렬을 만드는 메소드

    # freq들을 cos,sin으로 바꾸기
    cos, sin = freqs.cos(), freqs.sin()

    # FP32보다 덜 정확하지만 빠름
    cos, sin = cos.bfloat16(), sin.bfloat16()

    # Q,K 와 cos,sin이 곱해질때 차원을 맞추기위해 None을 추가해서 4차원으로 늘림. 
    # 1은 Broadcasting되어 자동 복사됨. Q, K: (B, T, n_head, head_dim) / cos, sin: (T = seq_len, head_dim)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]

    return cos, sin



  """
  (left, right) / left: 과거 몇칸? , right: 미래 몇칸?
  (-1, 0): 과거는 끝까지보고, 미래는 안봄: Causal Attention
  """
  # 이번 레이어의 window_size를 계산하는 함수가 아니라,
  # 모델 전체 레이어의 window_size를 미리 만들어두는 함수
  def _compute_window_sizes(self, config):

    pattern = config.window_pattern.upper()
    long_window = config.sequence_len
    short_window = long_window // 2
    char_to_window = {
        "L": (long_window, 0),
        "S": (short_window, 0)
    }

    window_sizes = []
    for layer_idx in range(config.n_layer):
      char = pattern[layer_idx % len(pattern)]
      window_sizes.append(char_to_window[char])

    window_sizes[-1] = (long_window, 0)
    return window_sizes

  def get_device(self):
      return self.transformer.wte.weight.device
  
  def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

  def num_scaling_params(self):
      """
      Return detailed parameter counts for scaling law analysis.
      Different papers use different conventions:
      - Kaplan et al. excluded embedding parameters
      - Chinchilla included all parameters
      Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
      Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

      Returns a dict with counts for each parameter group, so downstream analysis
      can experiment with which combination gives the cleanest scaling laws.
      """
      # Count each group separately (mirrors the grouping in setup_optimizers)
      wte = sum(p.numel() for p in self.transformer.wte.parameters())
      value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
      lm_head = sum(p.numel() for p in self.lm_head.parameters())
      transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
      scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
      total = wte + value_embeds + lm_head + transformer_matrices + scalars
      assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
      return {
          'wte': wte,
          'value_embeds': value_embeds,
          'lm_head': lm_head,
          'transformer_matrices': transformer_matrices,
          'scalars': scalars,
          'total': total,
      }

# MuonAdamW 옵티마이저가 먹기 좋게 데이터 형식과 내용 맞춰주기
  def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
    model_dim = self.config.n_embd
    # 분산학습인지 여부 / 전체 GPU중 나의 번호 / 현재 컴퓨터(노드)안에서의 번호 / 전체 GPU가 총 몇대인지 
    ddp, rank, local_rank, world_size = get_dist_info() 
    
    matrix_params = list(self.transformer.h.parameters())
    value_embeds_params = list(self.value_embeds.parameters())
    embedding_params = list(self.transformer.wte.parameters())
    lm_head_params = list(self.lm_head.parameters())
    resid_params = [self.resid_lambdas]
    x0_params = [self.x0_lambdas]
    assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

    dmodel_lr_scale = (model_dim / 768) ** -0.5 # lr 조절
    print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
    
    # parameter를 weights들 group별로 dictionary로 관리.
    # 우선 먼저 AdamW으로 계산할 parameter들 모음 
    param_groups = [
    dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0), 
    ]
    
    # Muon으로 계산할 parameter들 모음을 param_groups에 추가
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params, lr=matrix_lr, momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
        ))
    
    Factory = DistMuonAdamW if ddp else MuonAdamW
    optimizer = Factory(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer
  
  
  
  
  def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
    B, T = idx.size()
    
    # cos, sin표에서 지금 당장 필요한 부분만 가져오기
    assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
    assert idx.device == self.cos.device, f"RoPE and idx are on different devices: {idx.device} != {self.cos.device}"
    assert self.cos.dtype == torch.bfloat16, f"RoPE must be in bfloat16"
    T0 = 0 if kv_cache is None else kv_cache.get_pos() # get_pos: 행렬에 쌓인 데이터의 개수
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # cos, sin은 precompute 함수에서 4차원으로 늘렸음. 생략인덱싱 덕분에 뒤 차원들은 안써도됨.

    # Transformer의 몸통 (Trunk) - token embedding -> block
    x = self.transformer.wte(idx) # 단어 임베딩
    x = norm(x) # RMSnorm
    x0 = x # Residual Connection을 위해 저장
    for i, block in enumerate(self.transformer.h): # init에 보면 "h" : Block들 번호순 리스트
      x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
      ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
      x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
    x = norm(x)

    # lm_head (최종 출력층: n_embd -> vocab_size)
    softcap = 15 # logits의 최대/최소를 강제 제한
    logits = self.lm_head(x) # (B, T, padded_vocab_size)
    logits = logits[..., :self.config.vocab_size] # padded 된 빈칸들 잘라내기
    logits = logits.float() # BF16 -> FP32 : tanh나 loss 계산은 정확해야하니..
    logits = softcap * torch.tanh(logits / softcap) # Logit Softcapping: 부드럽게 -15~15 사이로 깎기
    
    # Loss 계산 
    if targets is not None: #Training mode 
      # 메모리가 많이드니까 문장을 chunk해보는 실험도 해볼만 할 듯함. // target이 -1인 부분은 무시
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)     
      return loss
    else: # inference mode 이므로 그냥 directly throw out 
      return logits


  @torch.inference_mode() # Gradient 계산 X 선언
  # tokens: 사용자 프롬프트 / top_k: 상위 k개 단어 필터링기능 옵션
  # Inference시엔 batch가 없지만 규격을 맞추기 위해 batch = 1이라하고 시작.
  def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
    assert isinstance(tokens, list)
    device = self.get_device()
    rng = None
    if temperature > 0:
      rng = torch.Generator(device=device)
      rng.manual_seed(seed)
    ids = torch.tensor([tokens], dtype=torch.long, device=device) # python list를 2차원 tensor로 변환
    for _ in range(max_tokens):
      logits = self.forward(ids) # (B, T, vocab_size)
      logits = logits[:, -1, :] # 
      if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # k가 logit size보다 작을수도있으니..
        logits[logits < v[:, [-1]]] = -float('Inf')
      if temperature > 0: 
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1, keepdim=True) # 1차원으로 눌러버리지 않고 (1, vocab_size)로 2차원을 유지함.
        next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
      else:
        next_ids = torch.argmax(logits, dim=-1, keepdim=True) # temperature가 0에 가까우면, 수치적 안정성을 위해 걍 argmax해라 
      ids = torch.cat((ids, next_ids), dim=1)
      token = next_ids.item() # item() : 텐서 안에 원소가 딱 1개만 있을때 Python 숫자로 꺼내오는 함수.
      yield token # Return인데, 함수 종료 안되고 계속 가는 Return
