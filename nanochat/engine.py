"""
用于高效推理我们模型的引擎。

整个系统围绕token序列工作：
- 用户可以向引擎发送token序列
- 引擎返回下一个token

注意事项：
- 引擎不了解分词，它纯粹处理token id序列。

整个系统尽可能高效。
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# 计算器工具辅助函数

@contextmanager
def timeout(duration, formula):
    """
    超时上下文管理器
    
    参数:
        duration: 超时时间（秒）
        formula: 计算公式
    """
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': 超时 {duration} 秒")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    """
    带超时的表达式计算函数
    
    参数:
        formula: 要计算的表达式
        max_time: 最大计算时间（秒）
    
    返回:
        计算结果或None（如果超时或出错）
    """
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula)
    except Exception as e:
        signal.alarm(0)
        # print(f"警告: 无法计算 {formula}, 异常: {e}") # 忽略错误的计算器使用是正常的
        return None

def use_calculator(expr):
    """
    安全地计算Python表达式。
    支持数学表达式和字符串操作，如 .count()
    
    参数:
        expr: 要计算的表达式字符串
    
    返回:
        计算结果或None（如果表达式不安全或无效）
    """
    # 移除数字中的逗号
    expr = expr.replace(",", "")

    # 检查是否为纯数学表达式（旧行为）
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # 不允许幂运算符
            return None
        return eval_with_timeout(expr)

    # 检查是否为我们支持的字符串操作
    # 允许: 字符串（单/双引号）、.count()、字母、数字、空格、括号
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # 禁止危险模式
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # 目前只允许 .count() 方法（以后可以扩展）
    if '.count(' not in expr:
        return None

    # 带超时计算
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    与GPT模型配合维护KV缓存。
    注意.pos在Transformer的最后一层插入后会自动前进。
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        """
        初始化KV缓存
        
        参数:
            batch_size: 批处理大小
            num_heads: 注意力头数
            seq_len: 序列长度
            head_dim: 每个头的维度
            num_layers: Transformer层数
        """
        # 每个K/V的形状为 (B, H, T, D)，Transformer的每一层都有一个
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0 # 缓存中的当前位置

    def reset(self):
        """重置缓存位置"""
        self.pos = 0

    def get_pos(self):
        """获取当前位置"""
        return self.pos

    def prefill(self, other):
        """
        给定另一个KV缓存进行预填充。可选择沿批处理维度扩展。
        当我们进行批处理大小为1的预填充，然后想要并行生成多个样本时使用。
        
        参数:
            other: 另一个KV缓存对象
        """
        # 1) 验证形状
        assert self.kv_cache is None, "无法预填充非空KV缓存"
        assert other.kv_cache is not None, "无法使用None KV缓存进行预填充"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                # num_layers, batch_size, num_heads, head_dim必须匹配
                assert dim1 == dim2, f"批处理维度不匹配: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size可以扩展
                assert dim1 == dim2 or dim2 == 1, f"批处理维度不匹配: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len: self必须比other长
                assert dim1 >= dim2, f"序列长度不匹配: {dim1} < {dim2}"
        # 2) 初始化缓存
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) 复制数据
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) 更新位置
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        """
        在指定层插入键值对
        
        参数:
            layer_idx: 层索引
            k: 键张量
            v: 值张量
        
        返回:
            更新后的键值视图
        """
        # 在此处惰性初始化缓存，因为我们需要知道数据类型/设备
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # 将新的键/值插入缓存并返回到目前为止的完整缓存
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # 如果需要，动态增长缓存
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # 所需的加上1024的缓冲区
            t_needed = (t_needed + 1023) & ~1023 # 然后向上舍入到1024的最近倍数
            current_shape = list(self.kv_cache.shape)
            current_shape[4] = t_needed
            self.kv_cache.resize_(current_shape)
        # 将k, v插入缓存
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # 返回当前位置之前的完整缓存键/值（作为视图）
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # 在Transformer的最后一层处理后递增pos
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """
    从给定的logits中采样单个下一个token
    
    参数:
        logits: 形状为(B, vocab_size)的logits张量
        rng: 随机数生成器
        temperature: 温度参数，控制采样随机性
        top_k: Top-K采样参数
    
    返回:
        形状为(B, 1)的下一个token张量
    """
    assert temperature >= 0.0, "温度必须为非负数"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    """行状态跟踪，用于生成过程中的状态管理"""
    def __init__(self, current_tokens=None):
        """
        初始化行状态
        
        参数:
            current_tokens: 当前行的token序列
        """
        self.current_tokens = current_tokens or [] # 当前行的token序列
        self.forced_tokens = deque() # 要强制注入的token队列
        self.in_python_block = False # 是否在python代码块内
        self.python_expr_tokens = [] # 当前python表达式的token
        self.completed = False # 该行是否已完成生成

class Engine:
    """模型推理引擎"""

    def __init__(self, model, tokenizer):
        """
        初始化引擎
        
        参数:
            model: 模型对象
            tokenizer: 分词器对象
        """
        self.model = model
        self.tokenizer = tokenizer # 工具使用需要

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        生成函数，执行单次预填充然后克隆KV缓存
        
        参数:
            tokens: 输入token列表
            num_samples: 生成样本数
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_k: Top-K采样参数
            seed: 随机种子
        
        生成:
            token列和对应的掩码
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "期望整数列表"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # 获取我们需要协调工具使用状态机的特殊token
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # 如果采样到，则结束行
        bos = self.tokenizer.get_bos_token_id() # 如果采样到，则结束行

        # 1) 对提示token进行批处理大小为1的预填充
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) 为每个样本/行复制KV缓存
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # 不需要保留这部分内存

        # 3) 为每个样本初始化状态
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) 主生成循环
        num_generated = 0
        first_iteration = True
        while True:
            # 停止条件：已达到最大token数
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # 停止条件：所有行都已完成
            if all(state.completed for state in row_states):
                break

            # 获取采样token - 来自预填充或前向传递
            if first_iteration:
                # 使用我们已经从预填充中采样的token
                sampled_tokens = [sampled_tokens[0]] * num_samples  # 将第一个token广播到所有行
                # TODO: 我们应该为每一行采样一个token而不是广播
                first_iteration = False
            else:
                # 前向传递模型并获取每行的下一个token
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size) 在最后时间步
                next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()

            # 处理每行：选择下一个token，更新状态，可选工具使用
            token_column = [] # 包含每行的下一个token id
            token_masks = [] # 包含掩码（是采样的(1)还是强制的(0)？）每行
            for i, state in enumerate(row_states):
                # 选择此行的下一个token
                is_forced = len(state.forced_tokens) > 0 # deque中是否有等待强制的token？
                token_masks.append(0 if is_forced else 1) # 如果是强制的掩码为0，如果是采样的掩码为1
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # 更新此行状态以包含下一个token
                state.current_tokens.append(next_token)
                # 在<|assistant_end|>或<|bos|>时，标记行已完成
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # 处理工具逻辑
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # 产出token列
            yield token_column, token_masks
            num_generated += 1
            # 为下一次迭代准备ids
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        非流式批处理生成，只返回最终的token序列。
        
        参数:
            tokens: 输入token列表
            num_samples: 生成样本数
            **kwargs: 其他参数
        
        返回:
            token序列列表（整数列表的列表）
            终止token（assistant_end, bos）不包含在结果中
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # 如果所有行都已完成则停止
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    快速内联测试，确保原始的model.generate函数
    与这里的快速Engine.generate函数等效。
    """
    import time
    # 初始化计算
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    # 加载模型和分词器
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # 通用超参数
    kwargs = dict(max_tokens=64, temperature=0.0)
    # 设置起始提示
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # 使用model.generate()函数生成参考序列
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"参考时间: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # 使用Engine生成token
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # 注意：在fp32中运行
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # 只打印第一行
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"引擎时间: {t1 - t0:.2f}s")
    # 比较两个序列
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"在 {i} 处不匹配: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"匹配: {reference_ids == generated_tokens}")