#!/usr/bin/env python3
"""
统一的网络聊天服务器 - 在单个FastAPI实例中同时提供UI界面和API接口。

使用数据并行方式将请求分发到多个GPU上处理。每个GPU加载完整的模型副本，
传入的请求会被分发到可用的工作进程上进行处理。

启动示例:

- 单个可用GPU（默认情况）
python -m scripts.chat_web

- 使用4个GPU
python -m scripts.chat_web --num-gpus 4

要开始聊天，请打开控制台中打印的URL。（如果在云服务器上，请确保使用公网IP）

接口列表:
  GET  /           - 聊天用户界面
  POST /chat/completions - 聊天API接口（仅支持流式响应）
  GET  /health     - 健康检查，包含工作池状态
  GET  /stats      - 工作池统计信息和GPU使用情况

防滥用机制:
  - 每个请求最多允许500条消息
  - 每条消息最多允许8000个字符
  - 整个对话长度最多允许32000个字符
  - 温度参数限制在0.0-2.0之间
  - Top-k参数限制在1-200之间
  - 最大令牌数限制在1-4096之间
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# 防滥用限制参数
# 每个请求最大消息数量
MAX_MESSAGES_PER_REQUEST = 500
# 每条消息最大字符数
MAX_MESSAGE_LENGTH = 8000
# 整个对话最大字符数
MAX_TOTAL_CONVERSATION_LENGTH = 32000
# 温度参数最小值
MIN_TEMPERATURE = 0.0
# 温度参数最大值
MAX_TEMPERATURE = 2.0
# Top-k参数最小值
MIN_TOP_K = 1
# Top-k参数最大值
MAX_TOP_K = 200
# 最大令牌数最小值
MIN_MAX_TOKENS = 1
# 最大令牌数最大值
MAX_MAX_TOKENS = 4096

# 命令行参数解析器，用于配置Web服务器
parser = argparse.ArgumentParser(description='NanoChat Web Server')
# 指定使用的GPU数量，默认为1
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
# 指定模型来源：sft(监督微调)|mid(中期训练)|rl(强化学习)，默认为sft
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
# 指定生成文本时的温度参数，默认为0.8
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
# 指定top-k采样参数，默认为50
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
# 指定生成的最大令牌数，默认为512
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
# 指定要加载的模型标签，默认为None
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
# 指定要加载的步骤，默认为None
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
# 指定服务器运行端口，默认为8000
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
# 指定数据类型：float32或bfloat16，默认为bfloat16
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
# 指定设备类型：cuda|cpu|mps，空值表示自动检测
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
# 指定服务器绑定的主机地址，默认为0.0.0.0
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
# 解析命令行参数
args = parser.parse_args()

# 配置日志记录，用于记录对话流量
logging.basicConfig(
    level=logging.INFO,                     # 设置日志级别为INFO
    format='%(asctime)s - %(message)s',     # 日志格式：时间戳 - 消息内容
    datefmt='%Y-%m-%d %H:%M:%S'             # 时间格式：年-月-日 时:分:秒
)
logger = logging.getLogger(__name__)        # 获取日志记录器实例

# 如果命令行未指定设备类型，则自动检测设备类型；否则使用指定的设备类型
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
# 初始化分布式训练相关参数和设备
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# 根据命令行参数设置PyTorch数据类型：float32或bfloat16
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

@dataclass
class Worker:
    """
    工作进程类，每个实例代表一个在特定GPU上加载模型的工作进程。
    
    属性:
        gpu_id: GPU标识符
        device: PyTorch设备对象
        engine: 模型引擎实例
        tokenizer: 分词器对象
        autocast_ctx: 自动混合精度上下文管理器
    """
    gpu_id: int                      # GPU编号
    device: torch.device            # 设备对象（如cuda:0）
    engine: Engine                  # 模型引擎
    tokenizer: object               # 分词器
    autocast_ctx: torch.amp.autocast # 自动混合精度上下文

class WorkerPool:
    """
    工作进程池类，管理多个工作进程，每个工作进程在不同的GPU上持有模型副本。
    
    该类负责初始化工作进程、管理工作进程的获取和释放。
    """

    def __init__(self, num_gpus: Optional[int] = None):
        """
        初始化工作进程池。
        
        参数:
            num_gpus: GPU数量，如果为None则自动检测
        """
        # 如果未指定GPU数量，则根据设备类型自动确定
        if num_gpus is None:
            if device_type == "cuda":
                # CUDA设备下获取可用GPU数量
                num_gpus = torch.cuda.device_count()
            else:
                # 非CUDA设备（如CPU或MPS）只使用1个进程
                num_gpus = 1 # e.g. cpu|mps
        self.num_gpus = num_gpus              # GPU数量
        self.workers: List[Worker] = []       # 工作进程列表
        # 可用工作进程队列，用于管理空闲的工作进程
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """
        在每个GPU上加载模型，初始化工作进程池。
        
        参数:
            source: 模型来源（sft|mid|rl）
            model_tag: 模型标签，可选
            step: 步骤编号，可选
        """
        print(f"正在初始化工作进程池，使用 {self.num_gpus} 个GPU...")
        # 如果使用多个GPU，必须是CUDA设备
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        # 遍历所有GPU，为每个GPU创建一个工作进程
        for gpu_id in range(self.num_gpus):

            # 根据设备类型设置设备对象
            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"正在GPU {gpu_id}上加载模型...")
            else:
                device = torch.device(device_type) # e.g. cpu|mps
                print(f"正在{device_type}上加载模型...")

            # 加载模型、分词器和配置
            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            # 创建引擎实例
            engine = Engine(model, tokenizer)
            # 创建自动混合精度上下文（仅CUDA设备使用）
            autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

            # 创建工作进程实例
            worker = Worker(
                gpu_id=gpu_id,           # GPU编号
                device=device,           # 设备对象
                engine=engine,           # 模型引擎
                tokenizer=tokenizer,     # 分词器
                autocast_ctx=autocast_ctx # 自动混合精度上下文
            )
            # 将工作进程添加到列表中
            self.workers.append(worker)
            # 将工作进程放入可用队列中
            await self.available_workers.put(worker)

        print(f"所有 {self.num_gpus} 个工作进程初始化完成!")

    async def acquire_worker(self) -> Worker:
        """
        从工作进程池中获取一个可用的工作进程。
        
        返回:
            Worker: 可用的工作进程实例
        """
        # 从可用工作进程队列中获取一个工作进程（如果队列为空会等待）
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """
        将工作进程归还到工作进程池中。
        
        参数:
            worker: 要归还的工作进程实例
        """
        # 将工作进程放回可用队列中
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    """
    聊天消息数据模型。
    
    属性:
        role: 消息角色（user/assistant）
        content: 消息内容
    """
    role: str      # 角色：user（用户）或assistant（助手）
    content: str   # 消息内容

class ChatRequest(BaseModel):
    """
    聊天请求数据模型。
    
    属性:
        messages: 消息列表
        temperature: 温度参数，可选
        max_tokens: 最大令牌数，可选
        top_k: Top-k采样参数，可选
    """
    messages: List[ChatMessage]     # 聊天消息列表
    temperature: Optional[float] = None  # 温度参数，控制生成文本的随机性
    max_tokens: Optional[int] = None     # 最大生成令牌数
    top_k: Optional[int] = None          # Top-k采样参数

def validate_chat_request(request: ChatRequest):
    """
    验证聊天请求参数，防止滥用。
    
    参数:
        request: 聊天请求对象
        
    异常:
        HTTPException: 当请求参数不符合要求时抛出HTTP异常
    """
    # 检查消息数量
    if len(request.messages) == 0:
        # 至少需要一条消息
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        # 消息数量不能超过最大限制
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
        )

    # 检查每条消息的长度和总对话长度
    total_length = 0
    for i, message in enumerate(request.messages):
        # 消息内容不能为空
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        # 检查单条消息长度是否超出限制
        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
            )
        # 累加总长度
        total_length += msg_length

    # 检查总对话长度是否超出限制
    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed"
        )

    # 验证角色值是否合法
    for i, message in enumerate(request.messages):
        # 角色只能是"user"或"assistant"
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user', 'assistant', or 'system'"
            )

    # 验证温度参数范围
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
            )

    # 验证top_k参数范围
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
            )

    # 验证max_tokens参数范围
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器，在应用启动时加载所有GPU上的模型。
    
    参数:
        app: FastAPI应用实例
    """
    print("正在所有GPU上加载nanochat模型...")
    # 创建工作进程池并保存到应用状态中
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    # 初始化工作进程池，加载模型
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"服务器已在 http://localhost:{args.port} 准备就绪")
    # 应用运行期间保持yield状态
    yield
    # 应用关闭时可以在这里添加清理代码（当前为空）

# 创建FastAPI应用实例，指定生命周期管理器
app = FastAPI(lifespan=lifespan)

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 允许所有来源
    allow_credentials=True,       # 允许携带凭证
    allow_methods=["*"],          # 允许所有HTTP方法
    allow_headers=["*"],          # 允许所有请求头
)

@app.get("/")
async def root():
    """
    根路径路由，提供聊天用户界面。
    
    返回:
        HTMLResponse: 包含聊天界面的HTML响应
    """
    # 构建UI HTML文件路径
    ui_html_path = os.path.join("nanochat", "ui.html")
    # 读取HTML文件内容
    with open(ui_html_path, "r") as f:
        html_content = f.read()
    # 替换API_URL为同源地址，使前端能够正确调用后端接口
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"  # 空字符串表示同源请求
    )
    # 返回HTML响应
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """
    Logo路由，提供NanoChat标志用于网站图标和页眉显示。
    
    返回:
        FileResponse: SVG格式的logo文件响应
    """
    # 构建logo文件路径
    logo_path = os.path.join("nanochat", "logo.svg")
    # 返回文件响应，指定媒体类型为SVG图像
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """
    流式生成助手响应。
    
    参数:
        worker: 工作进程实例
        tokens: 输入的令牌序列
        temperature: 温度参数，控制生成随机性
        max_new_tokens: 最大生成令牌数
        top_k: Top-k采样参数
        
    返回:
        AsyncGenerator[str, None]: 异步生成器，逐个产生生成的文本片段
    """
    # 如果未指定参数，则使用命令行默认值
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    # 获取特殊令牌ID
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")  # 助手结束标记
    bos = worker.tokenizer.get_bos_token_id()  # 开始标记

    # 累积令牌以正确处理多字节UTF-8字符（如表情符号）
    accumulated_tokens = []
    # 跟踪最后一个完整的UTF-8字符串（不包含替换字符）
    last_clean_text = ""

    # 使用自动混合精度上下文（如果支持）
    with worker.autocast_ctx:
        # 调用引擎生成文本
        for token_column, token_masks in worker.engine.generate(
            tokens,
            num_samples=1,           # 生成样本数
            max_tokens=max_new_tokens,  # 最大令牌数
            temperature=temperature,    # 温度参数
            top_k=top_k,               # Top-k参数
            seed=random.randint(0, 2**31 - 1)  # 随机种子
        ):
            # 获取生成的第一个令牌
            token = token_column[0]

            # 停止条件：遇到助手结束标记或开始标记
            if token == assistant_end or token == bos:
                break

            # 将令牌添加到序列中
            accumulated_tokens.append(token)
            # 解码所有累积的令牌以获得正确的UTF-8处理
            # 注意解码是一个相当高效的操作，基本上是查表和字符串连接
            current_text = worker.tokenizer.decode(accumulated_tokens)
            # 只有在文本不以替换字符结尾时才发送文本
            # 这确保我们不会发送不完整的UTF-8序列
            if not current_text.endswith(''):
                # 提取自上次完整解码以来的新文本
                new_text = current_text[len(last_clean_text):]
                if new_text:  # 只有在有新内容时才发送
                    # 以SSE格式发送数据
                    yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                    # 更新最后完整文本
                    last_clean_text = current_text

    # 发送完成信号
    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    聊天完成接口（仅支持流式响应）- 使用工作进程池实现多GPU支持。
    
    参数:
        request: 聊天请求对象
        
    返回:
        StreamingResponse: 流式响应对象
    """

    # 基本验证以防止滥用
    validate_chat_request(request)

    # 将传入的对话记录到控制台
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    # 从工作进程池中获取一个工作进程（如果都忙则等待）
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # 构建对话令牌序列
        bos = worker.tokenizer.get_bos_token_id()           # 开始标记
        user_start = worker.tokenizer.encode_special("<|user_start|>")    # 用户开始标记
        user_end = worker.tokenizer.encode_special("<|user_end|>")        # 用户结束标记
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")  # 助手开始标记
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")      # 助手结束标记

        # 初始化对话令牌序列，以开始标记开头
        conversation_tokens = [bos]
        # 遍历所有消息，构建完整的对话令牌序列
        for message in request.messages:
            if message.role == "user":
                # 用户消息：添加用户开始标记、消息内容、用户结束标记
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                # 助手消息：添加助手开始标记、消息内容、助手结束标记
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        # 添加助手开始标记，准备生成回复
        conversation_tokens.append(assistant_start)

        # 响应令牌列表，用于记录生成的响应内容
        response_tokens = []
        
        # 流式响应和释放工作进程的异步函数
        async def stream_and_release():
            try:
                # 异步迭代生成流
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # 累积响应内容用于日志记录
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    # 产生数据块
                    yield chunk
            finally:
                # 将助手响应记录到控制台
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # 流式传输完成后将工作进程释放回池中
                await worker_pool.release_worker(worker)

        # 返回流式响应
        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"  # 设置媒体类型为SSE
        )
    except Exception as e:
        # 确保即使在出错时也能释放工作进程
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """
    健康检查接口，返回服务器状态信息。
    
    返回:
        dict: 包含服务器状态、工作进程等信息的字典
    """
    # 获取工作进程池实例
    worker_pool = getattr(app.state, 'worker_pool', None)
    # 返回健康状态信息
    return {
        "status": "ok",                                    # 服务器状态：正常
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,  # 是否已准备好
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,             # GPU数量
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0  # 可用工作进程数
    }

@app.get("/stats")
async def stats():
    """
    获取工作进程池统计信息和GPU使用情况。
    
    返回:
        dict: 包含统计信息的字典
    """
    # 获取工作进程池实例
    worker_pool = app.state.worker_pool
    # 返回统计信息
    return {
        "total_workers": len(worker_pool.workers),                          # 总工作进程数
        "available_workers": worker_pool.available_workers.qsize(),         # 可用工作进程数
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),  # 忙碌工作进程数
        "workers": [                         # 各个工作进程详情
            {
                "gpu_id": w.gpu_id,          # GPU编号
                "device": str(w.device)      # 设备信息
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    # 导入uvicorn服务器
    import uvicorn
    # 打印服务器启动信息
    print(f"正在启动NanoChat Web服务器")
    # 打印生成参数配置
    print(f"温度参数: {args.temperature}, Top-k: {args.top_k}, 最大令牌数: {args.max_tokens}")
    # 运行uvicorn服务器
    uvicorn.run(app, host=args.host, port=args.port)
