"""行人疏散模拟系统常量定义模块"""

# ================================基本参数=============================== #
MAX_SPEED = 2.0  # Maximum speed of an individual
PANIC_THRESHOLD = 0.7  # Maximum mental stress one could bear
SIMULATION_DT = 0.1  # Interval between frames
ATTRACTION = 2.0  # Factor to acc_goal
HERD_FACT = 0.3  # C_int
HERD_FACT_NORMAL = 0.1  # C_int if not leader

# ================================行人类型参数=============================== #
DESIRED_SPEED = {
    "adult": 2.0,  # Desired speed of an adult (m/s)
    "elder": 1.3,  # an elderly
    "child": 1.3  # and an children
}

SAFE_DISTANCE = {
    "adult": 0.3,  # Safe distance from danger (m)
    "elder": 0.4,
    "child": 0.5
}

REACTION_TIME = {
    "adult": 0.5,  # How long he/she takes to adjust (s)
    "elder": 1.2,
    "child": 0.8
}

PERCEIVE_RADIUS = {
    "adult": 5.0,  # Which determines how stressed one feels
    "elder": 5.0,
    "child": 5.0
}

INTERPERSONAL_FORCE_PARAMS = {
    "adult": {"A": 1.0, "B": 0.8},  # Params when it comes to forces calc
    "elder": {"A": 1.2, "B": 1.0},
    "child": {"A": 1.5, "B": 1.0}
}

# ================================环境参数=============================== #
ENVIRONMENTAL_FORCE_PARAMS = {
    "danger": {"A": 5.0, "B": 0.8},  # Params uses when calc forces from dangers
    "obstacle": {"A": 5.0, "B": 2.0}
}

DANGER_EFFECT_PARAMS = {
    "default": {"lambda": 5.0, "stimuli": 1.0},  # Params for danger
    "fire": {"lambda": 0.5, "stimuli": 50.0}  # Params for fire (will be overridden by FIRE_DANGER_*)
}

# ================================路径选择参数=============================== #
HERD_BEHAVIOR_FACTOR = 0.5  # 从众行为因子（0-1，越高越从众）
STIMULUS_THRESHOLD = 0.8  # 刺激阈值，超过此值会随机选择
DANGER_WEIGHT = 2.0  # 危险程度权重

# ================================心理状态影响参数=============================== #
PANIC_PROBABILITY_MEAN = 1.0  # 正态分布概率因子的均值
PANIC_PROBABILITY_STD = 0.3  # 正态分布概率因子的标准差

# ================================火灾参数=============================== #
# 火灾扩散参数
FIRE_SPREAD_TIME_MIN = 2.0  # 火灾向相邻区域扩散的最小时间（秒）
FIRE_SPREAD_TIME_MAX = 5.0  # 火灾向相邻区域扩散的最大时间（秒）
FIRE_DURATION_MIN = 10.0  # 火灾持续的最小时间（秒）
FIRE_DURATION_MAX = 20.0  # 火灾持续的最大时间（秒）
FIRE_SPREAD_PROBABILITY = 0.6  # 火灾扩散到相邻区域的概率（0-1）

# 火灾危险参数
FIRE_DANGER_LAMBDA = 8.0  # 火灾危险影响范围参数（lambda）
FIRE_DANGER_STIMULI = 2.0  # 火灾基础刺激值（比普通危险源更高）
FIRE_ENVIRONMENTAL_FORCE_A = 20.0  # 火灾环境力参数A（比普通危险源更强，产生较大推力）
FIRE_ENVIRONMENTAL_FORCE_B = 0.8  # 火灾环境力参数B（较小的值使推力在更远距离仍然有效）

# 火灾生成参数
FIRE_INITIAL_COUNT = 0  # 初始火灾数量（0表示不自动生成）
FIRE_SPAWN_PROBABILITY = 0.0  # 每帧生成火灾的概率（0-1，0表示不自动生成）
FIRE_SPAWN_AREAS = []  # 可以生成火灾的区域ID列表（空列表表示所有可通行区域）

# ================================AI路径推荐参数=============================== #
# MPC（模型预测控制）参数
MPC_ENABLED = False  # 是否启用AI路径推荐
MPC_HORIZON = 5  # MPC预测时域（预测未来N步）
MPC_UPDATE_INTERVAL = 1.0  # AI推荐更新间隔（秒）
MPC_AI_RECOMMENDATION_WEIGHT = 0.3  # AI推荐权重（0-1，越高越信任AI）
MPC_DENSITY_WEIGHT = 0.5  # 人流密度权重（在MPC代价函数中）
MPC_DANGER_WEIGHT = 0.5  # 危险程度权重（在MPC代价函数中）
MPC_DISTANCE_WEIGHT = 0.2  # 距离权重（在MPC代价函数中）
