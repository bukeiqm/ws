# 行人疏散模拟系统

基于社会力模型的行人疏散模拟系统，支持心理状态影响、智能路径选择等功能。

## 项目结构

```
ws/
├── pedestrian_evacuation/          # 主包目录
│   ├── __init__.py                 # 包初始化，导出主要类和函数
│   ├── constants.py                # 所有常量定义
│   ├── entities.py                 # 数据实体类（仅存储数据）
│   ├── managers.py                 # 管理器类（处理业务逻辑）
│   ├── models.py                   # 模型类（Map等）
│   ├── monitor.py                  # 监视器模块
│   └── simulation.py               # 模拟和可视化代码
├── main.py                         # 主入口文件
├── README.md                       # 项目说明
└── ARCHITECTURE.md                 # 架构设计说明
```

## 模块说明

### constants.py
包含所有系统常量：
- 基本参数（最大速度、恐慌阈值、时间步长等）
- 行人类型参数（成人、老人、儿童的速度、反应时间等）
- 环境参数（障碍物、危险源的力参数）
- 路径选择参数
- 心理状态影响参数

### entities.py（数据实体类）
**仅存储数据，不包含业务逻辑**：
- `AgentData`: 行人数据类
- `ObstacleData`: 障碍物数据类
- `DangerData`: 危险源数据类
- `AreaData`: 区域数据类

### managers.py（管理器类）
**处理所有业务逻辑**：
- `AgentManager`: 管理Agent的状态更新
- `PhysicsEngine`: 处理物理计算（加速度、力等）
- `PathPlanner`: 处理路径规划和选择
- `EnvironmentManager`: 处理环境交互（障碍物、危险源）
- `BoundaryManager`: 处理边界力

### models.py
包含模型类：
- `Map`: 地图管理类，包含路径查找、区域连接等功能

### monitor.py
监视器模块：
- `AgentMonitor`: 行人监视器类，记录和显示特定行人的详细状态
- `AccelerationRecord`: 加速度记录数据类
- `PathSelectionWeights`: 路径选择权重数据类

### simulation.py
包含模拟和可视化相关代码：
- `run_simulation()`: 运行完整的仿真
- `plot_agent_parameters()`: 绘制行人参数随时间的变化
- `get_random_position_in_passable_area()`: 辅助函数，在可通行区域内随机生成位置

## 使用方法

### 基本使用

```python
from pedestrian_evacuation import run_simulation

# 运行仿真
ani = run_simulation(use_smart_choice=True)
```

### 自定义参数

```python
from pedestrian_evacuation import run_simulation

# 自定义参数运行
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    num_obstacles=15,
    num_danger=5
)
```

## 系统功能

### 1. 监视器系统

监视器系统可以实时监控特定行人的详细状态，并以图表形式显示数据随时间的变化。

#### 基本使用

```python
# 监视第一个行人（索引0），显示实时数据图表
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    monitor_agent_index=0  # 监视第一个行人
)
```

#### 监视器显示内容

监视器窗口包含4个图表：
- **恐慌值随时间变化**：红色曲线显示恐慌值变化
- **加速度分量随时间变化**：8条曲线显示各项加速度（目标驱动、行人间作用、障碍物、普通危险源、火灾、边界、心理因素、总加速度）
- **路径选择权重随时间变化**：5条曲线显示各项权重（距离、危险、障碍物、从众、AI推荐）
- **速度随时间变化**：蓝色曲线显示速度变化

#### 监视 + 其他系统

```python
# 监视行人 + 火灾系统
ani = run_simulation(
    num_agents=20,
    enable_fire=True,
    fire_initial_count=2,
    monitor_agent_index=0  # 观察火灾对行人的影响
)

# 监视行人 + AI路径推荐
ani = run_simulation(
    num_agents=20,
    enable_ai=True,
    ai_recommendation_weight=0.5,
    monitor_agent_index=0  # 观察AI推荐的影响
)
```

**注意事项**：
- `monitor_agent_index` 必须在有效范围内（0 到 num_agents-1）
- 监视器会增加少量计算开销，但影响很小
- 监视器窗口和主模拟窗口是独立的，可以分别关闭

### 2. 火灾系统

火灾系统使用**元胞自动机（Cellular Automaton）**模型来模拟火灾的扩散和熄灭过程。

#### 基本使用

```python
# 启用火灾系统
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    enable_fire=True  # 启用火灾系统
)
```

#### 自定义参数

```python
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    enable_fire=True,
    # 火灾参数
    fire_initial_count=2,              # 初始火灾数量
    fire_spawn_probability=0.001,      # 每帧生成火灾的概率
    fire_spread_time_min=2.0,          # 火灾扩散最小时间（秒）
    fire_spread_time_max=5.0,          # 火灾扩散最大时间（秒）
    fire_duration_min=10.0,            # 火灾持续最小时间（秒）
    fire_duration_max=20.0,            # 火灾持续最大时间（秒）
    fire_spread_probability=0.6         # 火灾扩散到相邻区域的概率
)
```

#### 参数说明

- **`fire_initial_count`** (默认: 0)：模拟开始时的初始火灾数量
- **`fire_spawn_probability`** (默认: 0.0)：每帧生成新火灾的概率（0-1）
- **`fire_spread_time_min/max`** (默认: 2.0/5.0秒)：火灾向相邻区域扩散的时间范围
- **`fire_duration_min/max`** (默认: 10.0/20.0秒)：火灾持续时间范围
- **`fire_spread_probability`** (默认: 0.6)：火灾扩散到相邻区域的概率

#### 火灾对行人的影响

1. **危险源影响**：火灾作为危险源，影响行人的路径选择
2. **环境力影响**：火灾产生环境力，推离行人
3. **恐慌值影响**：火灾会增加行人的恐慌值，影响其行为

**注意事项**：
- 火灾只能在**可通行区域**生成和扩散
- 火灾不会扩散到**出口区域**（避免阻塞逃生路径）

### 3. AI路径推荐系统

AI路径推荐系统使用**MPC（模型预测控制）**算法为行人提供最优路径推荐。

#### 基本使用

```python
# 启用AI路径推荐
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    enable_ai=True  # 启用AI路径推荐
)
```

#### 自定义参数

```python
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    enable_ai=True,
    # AI推荐参数
    ai_recommendation_weight=0.4,  # AI推荐权重（0-1）
    mpc_horizon=5,                 # MPC预测时域（预测未来N步）
    mpc_update_interval=1.0         # AI推荐更新间隔（秒）
)
```

#### 参数说明

- **`ai_recommendation_weight`** (默认: 0.3)：AI推荐在路径选择中的权重（0-1）
  - 值越大，行人越信任AI推荐
  - 建议值：0.2-0.6
- **`mpc_horizon`** (默认: 5)：MPC预测未来多少步
  - 值越大，预测越远，但计算开销也越大
  - 建议值：3-8
- **`mpc_update_interval`** (默认: 1.0秒)：AI推荐更新的时间间隔
  - 值越小，更新越频繁，但计算开销也越大
  - 建议值：0.5-2.0秒

#### AI + 火灾系统

```python
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    enable_fire=True,              # 启用火灾系统
    fire_initial_count=2,
    enable_ai=True,                # 启用AI路径推荐
    ai_recommendation_weight=0.5,  # AI会考虑火灾危险
    mpc_horizon=7                  # 更长的预测时域
)
```

#### MPC算法原理

MPC使用以下代价函数评估路径：
```
代价 = 距离代价 + 密度代价 + 危险代价
```

- **距离代价**：路径到出口的距离
- **密度代价**：预测路径上的人流密度
- **危险代价**：预测路径上的危险程度（包括火灾）

**注意事项**：
- 必须启用智能路径选择（`use_smart_choice=True`）
- 当行人恐慌值超过阈值时，会随机选择，忽略AI推荐
- AI推荐会缓存，根据更新间隔更新，减少计算开销

### 使用核心模型（新架构）

```python
from pedestrian_evacuation import AgentData, AgentManager, Map, ObstacleData, DangerData
from pedestrian_evacuation import constants

# 创建地图
map_manager = Map()
area_id = map_manager.add_node([0, 0], [10, 10])

# 创建数据对象
agent = AgentData([5, 5], agent_type="adult")
obstacle = ObstacleData([3, 3])
danger = DangerData([7, 7])

# 创建管理器
agent_manager = AgentManager(map_manager)

# 使用管理器更新状态
agent_manager.update_state(agent, neighbors=[], obstacles=[obstacle], 
                          dangers=[danger], boundary_area=None)
```

### 向后兼容（旧方式仍可用）

```python
from pedestrian_evacuation import Agent, Map, Obs, Dan  # 这些是别名

# 旧代码仍然可以工作，但建议迁移到新架构
agent = Agent([5, 5], agent_type="adult")
```

## 运行

```bash
python main.py
```

## 特性

- ✅ 基于社会力模型的行人动力学
- ✅ 智能路径选择（考虑距离、危险、障碍物、从众行为）
- ✅ 心理状态（panic）对加速度和路径选择的影响
- ✅ 正态分布概率因子调控心理状态影响
- ✅ A*路径规划算法
- ✅ 实时可视化
- ✅ **监视器系统**：实时监控行人状态，以图表形式显示数据随时间变化
- ✅ **火灾系统**：基于元胞自动机的火灾扩散和熄灭模拟
- ✅ **AI路径推荐**：基于MPC算法的智能路径推荐
- ✅ **数据与逻辑分离架构**：易于维护和扩展
- ✅ **清晰的状态管理**：通过管理器类统一管理

## 架构优势

新架构采用**数据与逻辑分离**的设计：

1. **数据类（entities.py）**：仅存储数据，不包含业务逻辑
2. **管理器类（managers.py）**：处理所有业务逻辑和状态管理
3. **单一职责**：每个管理器负责特定功能领域
4. **易于测试**：数据与逻辑分离使得单元测试更容易
5. **易于维护**：修改业务逻辑只需修改管理器类

详细说明请参考 [ARCHITECTURE.md](ARCHITECTURE.md)

