"""行人监视器模块 - 实时记录和显示特定行人的状态"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from .entities import AgentData


@dataclass
class AccelerationRecord:
    """加速度记录"""
    acc_goal: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 目标驱动加速度
    acc_int: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 行人间相互作用加速度
    acc_obstacle: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 障碍物加速度
    acc_danger: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 普通危险源加速度
    acc_fire: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 火灾加速度
    acc_bound: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 边界加速度
    acc_psy: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 心理因素加速度
    acc_total: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 总加速度


@dataclass
class PathSelectionWeights:
    """路径选择权重记录"""
    distance_weight: float = 0.0
    danger_weight: float = 0.0
    obstacle_weight: float = 0.0
    herd_weight: float = 0.0
    ai_weight: float = 0.0


class AgentMonitor:
    """行人监视器：记录特定行人的详细状态"""
    
    def __init__(self, agent: AgentData):
        self.agent = agent
        self.agent_id = id(agent)
        
        # 时间序列数据
        self.time_history: List[float] = []
        self.panic_history: List[float] = []
        self.speed_history: List[float] = []  # 速度历史
        
        # 加速度记录
        self.acceleration_history: List[AccelerationRecord] = []
        
        # 路径选择权重记录
        self.path_weights_history: List[PathSelectionWeights] = []
        
        # 当前值（用于实时显示）
        self.current_acceleration: Optional[AccelerationRecord] = None
        self.current_path_weights: Optional[PathSelectionWeights] = None
        self.current_panic: float = 0.0
    
    def record(
        self,
        current_time: float,
        acceleration: AccelerationRecord,
        path_weights: PathSelectionWeights
    ):
        """记录当前状态"""
        self.time_history.append(current_time)
        self.panic_history.append(self.agent.panic)
        self.speed_history.append(float(np.linalg.norm(self.agent.vel)))  # 记录速度大小
        self.acceleration_history.append(acceleration)
        self.path_weights_history.append(path_weights)
        
        # 更新当前值
        self.current_acceleration = acceleration
        self.current_path_weights = path_weights
        self.current_panic = self.agent.panic
    
    def get_acceleration_magnitudes(self) -> Dict[str, float]:
        """获取各项加速度的大小（模长）"""
        if self.current_acceleration is None:
            return {
                'goal': 0.0,
                'interpersonal': 0.0,
                'environmental': 0.0,  # 合并后的环境加速度
                'psychological': 0.0,
                'total': 0.0
            }
        
        acc = self.current_acceleration
        # 计算环境加速度（障碍物 + 危险源 + 火灾 + 边界）
        acc_environmental = (acc.acc_obstacle + acc.acc_danger + 
                            acc.acc_fire + acc.acc_bound)
        
        return {
            'goal': float(np.linalg.norm(acc.acc_goal)),
            'interpersonal': float(np.linalg.norm(acc.acc_int)),
            'environmental': float(np.linalg.norm(acc_environmental)),  # 合并后的环境加速度
            'psychological': float(np.linalg.norm(acc.acc_psy)),
            'total': float(np.linalg.norm(acc.acc_total))
        }
    
    def get_path_weights_dict(self) -> Dict[str, float]:
        """获取路径选择权重字典"""
        if self.current_path_weights is None:
            return {
                'distance': 0.0,
                'danger': 0.0,
                'obstacle': 0.0,
                'herd': 0.0,
                'ai': 0.0
            }
        
        weights = self.current_path_weights
        return {
            'distance': weights.distance_weight,
            'danger': weights.danger_weight,
            'obstacle': weights.obstacle_weight,
            'herd': weights.herd_weight,
            'ai': weights.ai_weight
        }
    
    def clear_history(self):
        """清除历史记录"""
        self.time_history.clear()
        self.panic_history.clear()
        self.speed_history.clear()
        self.acceleration_history.clear()
        self.path_weights_history.clear()
