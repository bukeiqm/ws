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

