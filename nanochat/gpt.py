"""
GPT 模型（重写版本，更加简化）
显著特性：
- 旋转位置编码（无传统的位置编码）
- QK 归一化
- 词元嵌入和语言模型头部使用不绑定的权重
- MLP 中使用 relu^2 激活函数
- 在词元嵌入后进行归一化
- rmsnorm 中无学习参数
- 线性层中无偏置
- 支持多查询注意力（MQA）以提高推理效率
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    """GPT 模型超参数配置类"""
    sequence_len: int = 1024      # 最大序列长度
    vocab_size: int = 50304       # 词汇表大小（填充至能被128整除）
    n_layer: int = 12             # Transformer 层数
    n_head: int = 6               # 查询头数量
    n_kv_head: int = 6            # 键/值头数量（MQA）
    n_embd: int = 768             # 嵌入维度


def norm(x):
    """
    纯函数式 rmsnorm，无学习参数。
    RMSNorm 是一种跨特征维度进行归一化的技术。
    
    参数：
        x: 输入张量，形状为 [..., feature_dim]
        
    返回：
        与输入形状相同的归一化张量
    """
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """
    对输入张量应用旋转位置编码。
    旋转编码通过旋转向量对的方式来编码位置信息。
    
    参数：
        x: 输入张量，形状为 [batch, heads, seq_len, head_dim]
        cos: 旋转编码的余弦分量
        sin: 旋转编码的正弦分量
        
    返回：
        应用了旋转编码的张量
    """
    assert x.ndim == 4  # 多头注意力
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # 将最后一个维度分成两半
    y1 = x1 * cos + x2 * sin # 旋转向量对
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # 重新组合
    out = out.to(x.dtype) # 确保输入/输出数据类型匹配
    return out


def repeat_kv(x, n_rep):
    """
    为多查询注意力复制键/值头。
    将 k/v 张量从 [batch, kv_heads, seq_len, head_dim] 扩展到 
    [batch, n_head, seq_len, head_dim]，通过在头维度上重复实现。
    
    参数：
        x: 输入张量，形状为 [batch, kv_heads, seq_len, head_dim]
        n_rep: 每个头重复的次数
        
    返回：
        扩展后的张量，具有重复的头
    """
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class CausalSelfAttention(nn.Module):
    """
    因果自注意力层，支持旋转编码和多查询注意力。
    实现注意力机制，其中每个词元只能关注序列中的前面词元。
    """
    
    def __init__(self, config, layer_idx):
        """
        初始化因果自注意力层。
        
        参数：
            config: 包含模型超参数的 GPTConfig 对象
            layer_idx: 此层在 transformer 中的索引
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        # 查询、键和值的线性投影
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # 输出投影
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        """
        因果自注意力的前向传播。
        
        参数：
            x: 输入张量，形状为 [batch, seq_len, n_embd]
            cos_sin: 余弦和正弦旋转编码的元组
            kv_cache: 用于高效推理的键值缓存（可选）
            
        返回：
            与输入形状相同的输出张量
        """
        B, T, C = x.size()

        # 投影输入以获得查询、键和值
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 对查询和键应用旋转嵌入以获得相对位置编码
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK 旋转嵌入
        q, k = norm(q), norm(k) # QK 归一化
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # 使头成为批次维度，即 (B, T, H, D) -> (B, H, T, D)

        # 应用 KV 缓存：将当前的 k,v 插入缓存，获取到目前为止的完整视图
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # 此次前向传递中查询的数量
        Tk = k.size(2) # 总共的键/值数量（缓存中的 + 当前前向传递的）

        # 应用 MQA：为每个查询头复制键/值头
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

        # 注意力：查询自回归地关注键/值。需要处理几种情况：
        if kv_cache is None or Tq == Tk:
            # 在训练期间（无 KV 缓存），像往常一样使用因果注意力
            # 即使有 KV 缓存，当 Tq == Tk 时我们仍可以使用这个简单版本
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # 在推理期间但此次前向传递只有一个查询：
            # 查询必须关注缓存中的所有键/值
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # 在推理期间且此次前向传递有一块查询：
            # 首先，每个查询关注所有缓存的键/值（即完整前缀）
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = 保留, False = 遮蔽
            prefix_len = Tk - Tq
            if prefix_len > 0: # 不能为负但可能为零
                attn_mask[:, :prefix_len] = True
            # 然后，在此块内进行因果注意力
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # 将各个头并排重新组装并投影回残差流
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    多层感知机，使用 ReLU^2 激活函数。
    一个简单的前馈网络，独立应用于每个词元。
    """
    
    def __init__(self, config):
        """
        初始化 MLP 层。
        
        参数：
            config: 包含模型超参数的 GPTConfig 对象
        """
        super().__init__()
        # 扩张层（4倍扩张是标准做法）
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        # 投影回原始维度
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        """
        MLP 的前向传播。
        
        参数：
            x: 输入张量，形状为 [batch, seq_len, n_embd]
            
        返回：
            与输入形状相同的输出张量
        """
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2 激活函数
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    Transformer 块，由注意力层和 MLP 层组成。
    在每个子层周围实现带有层归一化的残差连接。
    """
    
    def __init__(self, config, layer_idx):
        """
        初始化 Transformer 块。
        
        参数：
            config: 包含模型超参数的 GPTConfig 对象
            layer_idx: 此层在 transformer 中的索引
        """
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        """
        Transformer 块的前向传播。
        
        参数：
            x: 输入张量，形状为 [batch, seq_len, n_embd]
            cos_sin: 余弦和正弦旋转编码的元组
            kv_cache: 用于高效推理的键值缓存（可选）
            
        返回：
            与输入形状相同的输出张量
        """
        # 应用带残差连接的注意力
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        # 应用带残差连接的 MLP
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """
    完整的 GPT 模型实现，带有旋转编码和多查询注意力。
    
    模型包括：
    1. 词元嵌入层
    2. Transformer 块堆栈
    3. 语言建模头部
    """
    
    def __init__(self, config):
        """
        初始化 GPT 模型。
        
        参数：
            config: 包含模型超参数的 GPTConfig 对象
        """
        super().__init__()
        self.config = config
        
        # Transformer 主干：词元嵌入和 Transformer 块
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),  # 词元嵌入
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),  # Transformer 块
        })
        
        # 语言建模头部（与词元嵌入不绑定）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 为了支持元设备初始化，我们在这里初始化旋转嵌入，但它是虚假的
        # 对于 rotary_seq_len，这些旋转嵌入在内存中非常小/便宜，
        # 所以我们过度计算它们，但如果达到那个数量就断言失败。
        # 未来我们可以动态增长缓存，现在这样就可以了。
        self.rotary_seq_len = config.sequence_len * 10 # 10倍过度计算应该足够了，TODO 使其更好？
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False 意味着它不会保存到检查点中
        self.register_buffer("sin", sin, persistent=False)
        
        # 将嵌入从 fp32 转换为 bf16：优化器可以容忍，且节省内存：模型和激活中都节省
        self.transformer.wte.to(dtype=torch.bfloat16)

    def init_weights(self):
        """使用指定的初始化方案初始化模型权重。"""
        self.apply(self._init_weights)
        # 清零分类器权重
        torch.nn.init.zeros_(self.lm_head.weight)
        # 清零所有块中的 c_proj 权重
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # 初始化旋转嵌入
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        """
        为不同模块类型进行自定义权重初始化。
        
        参数：
            module: 要初始化的 PyTorch 模块
        """
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: 提高基础 theta，例如最近 100K 更常见
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """
        为所有位置预计算旋转嵌入，直到 seq_len。
        
        参数：
            seq_len: 最大序列长度
            head_dim: 每个注意力头的维度
            base: 频率计算的基础
            device: 创建嵌入的设备
            
        返回：
            (cos, sin) 旋转嵌入的元组
        """
        # 自动检测模型嵌入的设备
        if device is None:
            device = self.transformer.wte.weight.device
        # 步进通道
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 步进时间步
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 计算每个 (时间, 通道) 对的旋转频率
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # 保持为 bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # 添加批次和头维度以供后续广播
        return cos, sin

    def get_device(self):
        """获取模型所在的设备。"""
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ 
        返回模型每词元的估计 FLOPs。
        参考: https://arxiv.org/abs/2204.02311 
        
        返回：
            每词元的估计 FLOPs
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """
        为不同的参数组设置单独的优化器。
        对嵌入、矩阵和 lm_head 参数使用不同的学习率。
        
        参数：
            unembedding_lr: 语言建模头部的学习率
            embedding_lr: 词元嵌入的学习率
            matrix_lr: Transformer 矩阵的学习率
            weight_decay: 权重衰减系数
            
        返回：
            优化器列表
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # 将所有参数分为3组（矩阵、嵌入、lm_head）
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # 为嵌入和 lm_head 创建 AdamW 优化器
        # 按 ∝1/√dmodel 缩放 AdamW 参数的学习率（已经为 768 维模型调整了学习率）
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"按 ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f} 缩放 AdamW 参数的学习率")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # 为线性层创建 Muon 优化器
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # 将两个优化器合并到一个列表中
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """
        GPT 模型的前向传播。
        
        参数：
            idx: 输入词元索引，形状为 [batch, seq_len]
            targets: 训练的目标词元索引（可选）
            kv_cache: 用于高效推理的键值缓存（可选）
            loss_reduction: 损失计算的归约方法
            
        返回：
            如果提供了 targets：损失值
            否则：logits，形状为 [batch, seq_len, vocab_size]
        """
        B, T = idx.size()

        # 获取当前序列长度的旋转嵌入（它们的形状为 (1, seq_len, 1, head_dim)）
        assert T <= self.cos.size(1), f"序列长度超出了旋转嵌入缓存：{T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"旋转嵌入和 idx 在不同设备上：{idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "旋转嵌入必须是 bfloat16"
        # 如果存在 kv 缓存，我们需要将旋转嵌入偏移到缓存中的当前位置
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # 截断缓存到当前序列长度

        # 前向传播 Transformer 主干
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # 前向传播 lm_head（计算 logits）
        softcap = 15
        if targets is not None:
            # 训练模式：计算并返回损失
            # TODO: 尝试使用 Liger Kernels / 分块交叉熵等
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits 软帽
            logits = logits.float() # 对 logits 使用 tf32/fp32
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # 推理模式：计算并返回 logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits 软帽
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        简单的自回归流式推理。
        为了简单起见，我们假设：
        - 批次大小为 1
        - ids 和生成的词元是简单的 Python 列表和整数
        
        参数：
            tokens: 初始词元 ID 列表
            max_tokens: 要生成的最大词元数
            temperature: 采样温度（更高 = 更随机）
            top_k: Top-k 采样参数（None = 无 top-k）
            seed: 采样的随机种子
            
        产生：
            一次生成一个词元 ID
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # 添加批次维度
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token