"""
新的和升级的聊天模式，因为自上一个版本以来很多代码发生了变化。

目前仅支持单GPU运行:
python -m scripts.chat_cli -i mid
"""

# 导入必要的模块
import argparse                 # 命令行参数解析模块
import torch                   # PyTorch深度学习框架
from nanochat.common import compute_init, autodetect_device_type, print_banner  # 设备初始化和自动检测函数
from contextlib import nullcontext     # 空上下文管理器
from nanochat.engine import Engine     # 模型引擎类
from nanochat.checkpoint_manager import load_model  # 模型加载函数

print_banner()

# 创建命令行参数解析器，用于配置聊天程序
parser = argparse.ArgumentParser(description='Chat with the model')
# 指定模型来源：sft(监督微调)|mid(中期训练)|rl(强化学习)，默认为sft
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
# 指定要加载的模型标签，默认为None
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
# 指定要加载的步骤，默认为None
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
# 指定提示语，获取单次响应，默认为空
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
# 指定生成文本时的温度参数，默认为0.6
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
# 指定top-k采样参数，默认为50
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
# 指定设备类型：cuda|cpu|mps，空值表示自动检测
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
# 指定数据类型：float32或bfloat16，默认为bfloat16
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
# 解析命令行参数
args = parser.parse_args()

# 初始化模型和分词器

# 如果命令行未指定设备类型，则自动检测设备类型；否则使用指定的设备类型
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
# 初始化分布式训练相关参数和设备
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# 根据命令行参数设置PyTorch数据类型：float32或bfloat16
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
# 创建自动混合精度上下文（仅CUDA设备使用）
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
# 加载模型、分词器和元数据
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# 聊天状态机的特殊令牌
# 获取开始标记(Beginning of Sentence)的ID
bos = tokenizer.get_bos_token_id()
# 获取用户消息开始和结束标记
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
# 获取助手消息开始和结束标记
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# 创建引擎以实现高效生成
engine = Engine(model, tokenizer)

# 打印欢迎信息和使用说明
print("\nNanoChat Interactive Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")  # 输入'quit'或'exit'结束对话
print("Type 'clear' to start a new conversation")      # 输入'clear'开始新对话
print("-" * 50)

# 初始化对话令牌序列，以开始标记开头
conversation_tokens = [bos]

# 进入主循环
while True:

    # 判断是否使用命令行提供的提示语
    if args.prompt:
        # 从启动命令获取提示语
        user_input = args.prompt
    else:
        # 从控制台交互式获取提示语
        try:
            user_input = input("\nUser: ").strip()  # 获取用户输入并去除首尾空格
        except (EOFError, KeyboardInterrupt):       # 处理EOF(Ctrl+D)或中断(Ctrl+C)异常
            print("\nGoodbye!")                     # 打印告别信息
            break                                   # 退出循环

    # 处理特殊命令
    if user_input.lower() in ['quit', 'exit']:      # 用户输入quit或exit
        print("Goodbye!")                           # 打印告别信息
        break                                       # 退出循环

    if user_input.lower() == 'clear':               # 用户输入clear
        conversation_tokens = [bos]                 # 重置对话令牌序列
        print("Conversation cleared.")              # 提示对话已清除
        continue                                    # 继续下一次循环

    if not user_input:                              # 用户输入为空
        continue                                    # 继续下一次循环

    # 将用户消息添加到对话中
    conversation_tokens.append(user_start)              # 添加用户消息开始标记
    conversation_tokens.extend(tokenizer.encode(user_input))  # 添加用户消息内容的编码
    conversation_tokens.append(user_end)                # 添加用户消息结束标记

    # 启动助手回复
    conversation_tokens.append(assistant_start)         # 添加助手消息开始标记
    # 设置生成参数
    generate_kwargs = {
        "num_samples": 1,           # 生成样本数为1
        "max_tokens": 256,          # 最大生成令牌数为256
        "temperature": args.temperature,  # 使用命令行指定的温度参数
        "top_k": args.top_k,        # 使用命令行指定的top-k参数
    }
    response_tokens = []            # 初始化响应令牌列表
    print("\nAssistant: ", end="", flush=True)  # 打印"Assistant: "前缀，不换行并立即刷新输出
    # 使用自动混合精度上下文进行生成
    with autocast_ctx:
        # 调用引擎生成文本
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0] # 获取批次维度的第一个令牌(num_samples=1)
            response_tokens.append(token)     # 将生成的令牌添加到响应列表
            token_text = tokenizer.decode([token])  # 解码令牌为文本
            print(token_text, end="", flush=True)   # 打印文本，不换行并立即刷新输出
    print()  # 换行
    
    # 我们必须确保助手结束令牌是最后一个令牌
    # 因此即使由于达到最大令牌数而结束生成，我们也必须将其附加到末尾
    if response_tokens[-1] != assistant_end:    # 如果最后一个令牌不是助手结束标记
        response_tokens.append(assistant_end)   # 添加助手结束标记
    conversation_tokens.extend(response_tokens) # 将响应令牌扩展到对话令牌序列中

    # 在提示模式下，我们只想要一次响应然后退出
    if args.prompt:    # 如果使用了命令行提示语
        break          # 退出循环
