# 在文件顶部添加导入
import sys
import os
ppdpp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'PPDPP')
if ppdpp_path not in sys.path:
    sys.path.insert(0, ppdpp_path)

from PPDPP.agent import PPDPP
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig
import torch

# 全局变量：缓存加载的模型（避免重复加载）
_ppdpp_policy = None
_ppdpp_args = None

def _init_ppdpp_policy():
    """初始化 PPDPP 策略网络（只加载一次）"""
    global _ppdpp_policy, _ppdpp_args
    
    if _ppdpp_policy is not None:
        return _ppdpp_policy, _ppdpp_args
    
    # ===== 参数写死 =====
    from argparse import Namespace
    args = Namespace()
    args.data_name = 'p4g'
    args.model_name = 'roberta'
    args.model_name_or_path = './PPDPP/roberta-large'
    args.cache_dir = './PPDPP/cache/plm'
    args.max_seq_length = 512
    args.learning_rate = 1e-6
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # SFT 和 RL 模型路径（根据实际情况修改）
    sft_dir = "./PPDPP/sft_outputs/p4g/roberta/best_checkpoint"
    load_rl_epoch = 7
    
    # 初始化模型和tokenizer
    tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
    cfg = {'bert': BertConfig, 'roberta': RobertaConfig}
    
    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    
    # 创建策略网络
    policy = PPDPP(args, config, tokenizer)
    
    # 加载模型
    if sft_dir and os.path.exists(sft_dir):
        print(f'[PPDPP] 加载 SFT 模型从 {sft_dir}')
        policy.load_model(data_name='p4g', filename=sft_dir)
    
    if load_rl_epoch > 0:
        filename = f'p4g-sft_outputs-chatgpt-chatgpt-chatgpt'
        print(f'[PPDPP] 加载 RL 模型 epoch {load_rl_epoch}')
        policy.load_model(data_name='p4g', filename=filename, epoch_user=load_rl_epoch)
    
    _ppdpp_policy = policy
    _ppdpp_args = args
    
    return policy, args

def select_strategy_with_ppdpp(dialog_history: List[Dict[str, Any]]) -> str:
    """
    使用 PPDPP 策略网络选择策略
    
    Args:
        dialog_history: 对话历史，格式为 [{"turn_id": 0, "speaker": "Persuader", "text": "..."}, ...]
        
    Returns:
        str: 选择的策略名称（如 "Logical appeal", "Emotion appeal" 等）
    """
    # 初始化模型（如果还没加载）
    policy, args = _init_ppdpp_policy()
    
    # ===== 转换数据格式 =====
    # 将 dialog_history 转换为 PPDPP 能理解的 state 格式
    # PPDPP 期望的格式: [{"role": "Persuader", "content": "..."}, {"role": "Persuadee", "content": "..."}, ...]
    state = []
    for turn in dialog_history:
        speaker = turn.get('speaker', '')
        text = turn.get('text', '')
        
        # 转换 speaker 为 role
        if speaker == 'Persuader':
            role = 'Persuader'
        elif speaker == 'Persuadee':
            role = 'Persuadee'
        else:
            role = speaker  # 保持原样
        
        state.append({
            'role': role,
            'content': text
        })
    
    # ===== 选择策略 =====
    action = policy.select_action(state, is_test=True)
    
    return action  # 返回策略名称，如 "Logical appeal"