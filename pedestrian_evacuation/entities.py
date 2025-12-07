"""行人疏散模拟系统 - 数据实体类（仅存储数据）"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ObstacleData:
    """障碍物数据类（仅存储数据）"""
    pos: np.ndarray
    obs_type: str = "obstacle"
    A: float = field(init=False)  # 将在初始化后设置
    B: float = field(init=False)
    
    def __post_init__(self):
        """初始化后设置参数"""
        from . import constants
        if isinstance(self.pos, (list, tuple)):
            self.pos = np.array(self.pos, dtype=float)
        self.A = constants.ENVIRONMENTAL_FORCE_PARAMS[self.obs_type]["A"]
        self.B = constants.ENVIRONMENTAL_FORCE_PARAMS[self.obs_type]["B"]


@dataclass
class DangerData:
    """危险源数据类（仅存储数据）"""
    pos: np.ndarray
    dan_type: str = "default"
    A: float = field(init=False)
    B: float = field(init=False)
    L: float = field(init=False)
    S: float = field(init=False)
    
    def __post_init__(self):
        """初始化后设置参数"""
        from . import constants
        if isinstance(self.pos, (list, tuple)):
            self.pos = np.array(self.pos, dtype=float)
        self.A = constants.ENVIRONMENTAL_FORCE_PARAMS["danger"]["A"]
        self.B = constants.ENVIRONMENTAL_FORCE_PARAMS["danger"]["B"]
        self.L = constants.DANGER_EFFECT_PARAMS[self.dan_type]["lambda"]
        self.S = constants.DANGER_EFFECT_PARAMS[self.dan_type]["stimuli"]


@dataclass
class FireData:
    """火灾数据类（仅存储数据）"""
    area_id: int  # 火灾所在的区域ID
    start_time: float  # 火灾开始时间
    spread_time: float  # 扩散到相邻区域的时间（随机生成）
    duration: float  # 火灾持续时间（随机生成）
    intensity: float = 1.0  # 火灾强度（0-1，可用于可视化）
    
    # 扩散相关
    next_spread_time: float = field(init=False)  # 下次扩散的时间
    spread_targets: List[int] = field(default_factory=list)  # 已标记要扩散的目标区域
    
    def __post_init__(self):
        """初始化后设置参数"""
        from . import constants
        import random
        # 计算下次扩散时间
        self.next_spread_time = (self.start_time + 
                                random.uniform(constants.FIRE_SPREAD_TIME_MIN, 
                                             constants.FIRE_SPREAD_TIME_MAX))


@dataclass
class AreaData:
    """区域数据类（仅存储数据）"""
    left_bottom: List[float]
    right_top: List[float]
    id: int
    property: int  # 0=道路, 1=路口
    jammed: List[str] = field(default_factory=list)  # 被阻塞的边界
    adjacent: List[int] = field(default_factory=list)  # 相邻区域ID
    
    @property
    def center(self):
        """区域中心点"""
        return [(self.left_bottom[0] + self.right_top[0]) / 2,
                (self.left_bottom[1] + self.right_top[1]) / 2]
    
    def inside(self, pos) -> bool:
        """判断点是否在区域内"""
        if isinstance(pos, np.ndarray):
            x, y = pos[0], pos[1]
        else:
            x, y = pos
        left, bottom = self.left_bottom
        right, top = self.right_top
        return left < x < right and bottom < y < top
    
    def is_node(self) -> bool:
        """判断是否为路口"""
        return self.property == 1
    
    def is_road(self) -> bool:
        """判断是否为道路"""
        return self.property == 0


@dataclass
class AgentData:
    """行人（智能体）数据类（仅存储数据）"""
    pos: np.ndarray
    agent_type: str = "adult"
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    panic: float = 0.0
    evacuated: bool = False
    
    # 路径相关
    current_area_id: int = -1
    exit_node_id: int = -1
    exit_pos: Optional[np.ndarray] = None
    target_pos: Optional[np.ndarray] = None
    path: List[int] = field(default_factory=list)
    path_index: int = 0
    next_target_area_id: int = -1
    use_smart_choice: bool = True
    
    # 心理状态
    panic_probability_factor: float = 1.0
    
    # 从常量中获取的参数（这些是只读的，基于agent_type）
    _desired_speed: float = field(init=False)
    _reaction_time: float = field(init=False)
    _safe_distance: float = field(init=False)
    _perceive_radius: float = field(init=False)
    _A: float = field(init=False)  # 人际力参数A
    _B: float = field(init=False)  # 人际力参数B
    _C: float = field(init=False)  # 从众因子
    
    def __post_init__(self):
        """初始化后设置参数"""
        from . import constants
        if isinstance(self.pos, (list, tuple)):
            self.pos = np.array(self.pos, dtype=float)
        if self.exit_pos is not None and isinstance(self.exit_pos, (list, tuple)):
            self.exit_pos = np.array(self.exit_pos, dtype=float)
        if self.target_pos is not None and isinstance(self.target_pos, (list, tuple)):
            self.target_pos = np.array(self.target_pos, dtype=float)
        
        # 从常量中获取参数
        self._desired_speed = constants.DESIRED_SPEED[self.agent_type]
        self._reaction_time = constants.REACTION_TIME[self.agent_type]
        self._safe_distance = constants.SAFE_DISTANCE[self.agent_type]
        self._perceive_radius = constants.PERCEIVE_RADIUS[self.agent_type]
        self._A = constants.INTERPERSONAL_FORCE_PARAMS[self.agent_type]["A"]
        self._B = constants.INTERPERSONAL_FORCE_PARAMS[self.agent_type]["B"]
        self._C = constants.HERD_FACT if self.agent_type == "adult" else constants.HERD_FACT_NORMAL
    
    @property
    def desired_speed(self) -> float:
        return self._desired_speed
    
    @property
    def reaction_time(self) -> float:
        return self._reaction_time
    
    @property
    def safe_distance(self) -> float:
        return self._safe_distance
    
    @property
    def perceive_radius(self) -> float:
        return self._perceive_radius
    
    @property
    def A(self) -> float:
        return self._A
    
    @property
    def B(self) -> float:
        return self._B
    
    @property
    def C(self) -> float:
        return self._C
