"""
Common utilities for nanochat.
nanochat项目的通用工具模块
"""

import os
import re
import logging
import fcntl
import urllib.request
import torch
import torch.distributed as dist

class ColoredFormatter(logging.Formatter):
    """自定义日志格式化器，为不同级别的日志消息添加颜色"""
    
    # ANSI颜色代码定义
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫红色
    }
    RESET = '\033[0m'  # 重置颜色
    BOLD = '\033[1m'   # 粗体
    
    def format(self, record):
        """
        格式化日志记录，为不同级别的日志添加颜色
        
        Args:
            record: 日志记录对象
            
        Returns:
            str: 格式化后的日志消息
        """
        # 为日志级别名称添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        
        # 格式化消息
        message = super().format(record)
        
        # 为INFO级别的消息中的数字和百分比添加高亮
        if levelname == 'INFO':
            # 高亮显示数字和单位(Gb, MB, %, docs)
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
            
        return message

def setup_default_logging():
    """
    设置默认的日志配置
    创建一个带颜色的日志处理器并设置基本的日志配置
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为INFO
        handlers=[handler]   # 使用自定义的带颜色格式化器
    )

# 初始化日志配置
setup_default_logging()
logger = logging.getLogger(__name__)  # 获取当前模块的logger实例

def get_base_dir():
    """
    获取nanochat的基础目录路径
    如果设置了NANOCHAT_BASE_DIR环境变量，则使用该路径
    否则，默认使用用户主目录下的.cache/nanochat路径
    
    Returns:
        str: nanochat基础目录的绝对路径
    """
    # 检查是否设置了NANOCHAT_BASE_DIR环境变量
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        # 默认使用~/.cache/nanochat目录
        home_dir = os.path.expanduser("~")      # 获取用户主目录
        cache_dir = os.path.join(home_dir, ".cache")  # 构造缓存目录路径
        nanochat_dir = os.path.join(cache_dir, "nanochat")  # 构造nanochat目录路径
    
    # 确保目录存在，如果不存在则创建
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename):
    """
    从URL下载文件到基础目录中的指定文件
    使用锁文件防止多个进程同时下载同一文件
    
    Args:
        url (str): 要下载文件的URL地址
        filename (str): 保存文件的名称
        
    Returns:
        str: 下载文件的完整路径
    """
    base_dir = get_base_dir()                           # 获取基础目录
    file_path = os.path.join(base_dir, filename)       # 构造文件完整路径
    lock_path = file_path + ".lock"                    # 构造锁文件路径

    # 如果文件已经存在，直接返回文件路径
    if os.path.exists(file_path):
        return file_path

    # 创建并获取锁文件
    with open(lock_path, 'w') as lock_file:
        # 只有一个进程可以获得此锁，其他进程会阻塞直到锁被释放
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # 再次检查文件是否存在（可能其他进程已完成下载）
        if os.path.exists(file_path):
            return file_path

        print(f"正在下载 {url}...")
        # 从URL获取内容
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')

        # 将内容写入文件
        with open(file_path, 'w') as f:
            f.write(content)

        print(f"已下载到 {file_path}")

    # 释放锁后清理锁文件
    try:
        os.remove(lock_path)
    except OSError:
        pass  # 忽略其他进程已删除的情况

    return file_path

def print0(s="", **kwargs):
    """
    只在主进程中打印消息的函数
    在分布式训练环境中，只有rank为0的进程会输出消息
    
    Args:
        s (str): 要打印的字符串
        **kwargs: 传递给print函数的其他参数
    """
    ddp_rank = int(os.environ.get('RANK', 0))  # 获取当前进程的rank，默认为0
    if ddp_rank == 0:  # 只有主进程(rank=0)才打印
        print(s, **kwargs)

def print_banner():
    """打印项目ASCII艺术横幅"""
    # 使用DOS Rebel字体创建的ASCII艺术横幅
    banner = """
                                                   █████                 █████
                                                  ░░███                 ░░███
 ████████    ██████   ████████    ██████   ██████  ░███████    ██████   ███████
░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███ ░░░███░
 ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████   ░███
 ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███   ░███ ███
 ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░████████  ░░█████
░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░    ░░░░░
"""
    print0(banner)  # 只在主进程打印横幅

def is_ddp():
    """
    检查是否使用了分布式数据并行(DDP)
    
    Returns:
        bool: 如果使用了DDP返回True，否则返回False
    """
    # TODO 是否有更合适的方法判断？
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    """
    获取分布式训练的相关信息
    
    Returns:
        tuple: (ddp, ddp_rank, ddp_local_rank, ddp_world_size)
               - ddp: 是否使用分布式训练
               - ddp_rank: 当前进程的全局rank
               - ddp_local_rank: 当前节点内的local rank
               - ddp_world_size: 总进程数
    """
    if is_ddp():
        # 检查必要的环境变量是否存在
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])              # 全局rank
        ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 本地rank
        ddp_world_size = int(os.environ['WORLD_SIZE'])  # 总进程数
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        # 非分布式环境，返回默认值
        return False, 0, 0, 1

def autodetect_device_type():
    """
    自动检测设备类型
    优先使用CUDA(如果有GPU)，其次使用MPS(苹果设备)，最后回退到CPU
    
    Returns:
        str: 检测到的设备类型 ("cuda"|"mps"|"cpu")
    """
    # 按优先级检测设备类型
    if torch.cuda.is_available():
        device_type = "cuda"     # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device_type = "mps"      # Apple Silicon GPU
    else:
        device_type = "cpu"      # CPU回退选项
    
    print0(f"自动检测到的设备类型: {device_type}")
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """
    基础初始化函数，包含项目中常用的初始化操作
    
    Args:
        device_type (str): 设备类型，可选值为"cuda"|"mps"|"cpu"，默认为"cuda"
        
    Returns:
        tuple: (ddp, ddp_rank, ddp_local_rank, ddp_world_size, device)
               - ddp: 是否使用分布式训练
               - ddp_rank: 当前进程的全局rank
               - ddp_local_rank: 当前节点内的local rank
               - ddp_world_size: 总进程数
               - device: PyTorch设备对象
    """

    # 验证设备类型参数的有效性
    assert device_type in ["cuda", "mps", "cpu"], "无效的设备类型"
    
    # 检查PyTorch是否支持所选的设备类型
    if device_type == "cuda":
        assert torch.cuda.is_available(), "PyTorch未配置CUDA支持但设备类型设置为'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "PyTorch未配置MPS支持但设备类型设置为'mps'"

    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # 暂时跳过完全的可重现性设置，后续可以研究是否影响性能
    # torch.use_deterministic_algorithms(True)

    # 设置浮点运算精度
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") # 使用tf32代替fp32进行矩阵乘法以提高性能

    # 分布式设置: 分布式数据并行(DDP)，可选，需要CUDA支持
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    
    if ddp and device_type == "cuda":
        # 在分布式CUDA环境中设置设备
        device = torch.device("cuda", ddp_local_rank)  # 创建指定GPU的设备对象
        torch.cuda.set_device(device)                  # 设置默认CUDA设备
        dist.init_process_group(backend="nccl", device_id=device)  # 初始化进程组
        dist.barrier()                                 # 同步所有进程
    else:
        # 非分布式或非CUDA环境
        device = torch.device(device_type)             # 创建设备对象(mps|cpu)

    # 主进程记录分布式世界大小信息
    if ddp_rank == 0:
        logger.info(f"分布式世界大小: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """
    与compute_init配套的清理函数，在脚本退出前清理资源
    销毁分布式进程组(如果使用了分布式训练)
    """
    if is_ddp():
        dist.destroy_process_group()

class DummyWandb:
    """
    虚拟Wandb类，当我们不想使用wandb但仍需要保持相同接口签名时非常有用
    提供与真实Wandb相同的接口但不执行任何实际操作
    """
    
    def __init__(self):
        """初始化虚拟Wandb实例"""
        pass
    
    def log(self, *args, **kwargs):
        """
        虚拟的日志记录方法，不执行任何操作
        保持与真实Wandb.log方法相同的签名
        """
        pass
    
    def finish(self):
        """
        虚拟的结束方法，不执行任何操作
        保持与真实Wandb.finish方法相同的签名
        """
        pass