"""行人疏散模拟系统

一个基于社会力模型的行人疏散模拟系统，支持心理状态影响、智能路径选择等功能。
"""

# 数据实体类
from .entities import AgentData, ObstacleData, DangerData, AreaData, FireData

# 管理器类
from .managers import (
    AgentManager,
    PhysicsEngine,
    PathPlanner,
    EnvironmentManager,
    BoundaryManager,
    FireManager,
    MPCPlanner
)

# 模型类（保留向后兼容）
from .models import Map

# 为了向后兼容，保留旧名称作为别名
from .entities import AgentData as Agent, ObstacleData as Obs, DangerData as Dan
Area = AreaData

# 模拟函数
from .simulation import run_simulation, plot_agent_parameters, get_random_position_in_passable_area

# 监视器模块
from .monitor import AgentMonitor, AccelerationRecord, PathSelectionWeights

# 常量模块
from . import constants

__all__ = [
    # 数据实体类（推荐使用）
    'AgentData',
    'ObstacleData',
    'DangerData',
    'AreaData',
    'FireData',
    # 管理器类
    'AgentManager',
    'PhysicsEngine',
    'PathPlanner',
    'EnvironmentManager',
    'BoundaryManager',
    'FireManager',
    'MPCPlanner',
    # 模型类
    'Map',
    # 向后兼容的别名
    'Agent',
    'Obs',
    'Dan',
    'Area',
    # 模拟函数
    'run_simulation',
    'plot_agent_parameters',
    'get_random_position_in_passable_area',
    # 监视器类
    'AgentMonitor',
    'AccelerationRecord',
    'PathSelectionWeights',
    # 常量模块
    'constants',
]

__version__ = '2.0.0'
