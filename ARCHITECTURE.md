# 项目架构说明

## 架构设计原则

本项目采用**数据与逻辑分离**的架构设计，遵循以下原则：

1. **数据类（Entities）**：仅存储数据，不包含业务逻辑
2. **管理器类（Managers）**：处理所有业务逻辑和状态管理
3. **单一职责**：每个管理器类负责特定的功能领域
4. **易于测试**：数据与逻辑分离使得单元测试更容易

## 项目结构

```
pedestrian_evacuation/
├── entities.py          # 数据实体类（仅存储数据）
├── managers.py          # 管理器类（处理业务逻辑）
├── models.py            # 模型类（Map等）
├── constants.py         # 常量定义
├── simulation.py        # 模拟和可视化
└── __init__.py          # 包初始化
```

## 数据实体类（entities.py）

### AgentData
- **职责**：仅存储行人的数据
- **包含**：位置、速度、恐慌值、路径信息等
- **不包含**：任何计算方法或业务逻辑

### ObstacleData / DangerData
- **职责**：仅存储障碍物/危险源的数据
- **包含**：位置、类型、参数等
- **不包含**：力计算等逻辑

### AreaData
- **职责**：仅存储区域的数据
- **包含**：边界、中心点、相邻关系等
- **不包含**：边界力计算等逻辑

## 管理器类（managers.py）

### AgentManager
- **职责**：管理Agent的状态更新
- **方法**：
  - `update_state()`: 更新Agent的完整状态
  - `update_panic()`: 更新panic值

### PhysicsEngine
- **职责**：处理所有物理计算
- **方法**：
  - `calculate_total_acceleration()`: 计算总加速度
  - `calculate_goal_acceleration()`: 计算目标加速度
  - `calculate_interpersonal_acceleration()`: 计算人际加速度
  - `calculate_psychological_acceleration()`: 计算心理加速度

### PathPlanner
- **职责**：处理路径规划和选择
- **方法**：
  - `find_path()`: A*路径查找
  - `choose_path()`: 智能路径选择
  - `update_path()`: 更新路径规划

### EnvironmentManager
- **职责**：处理环境交互（障碍物、危险源）
- **方法**：
  - `calculate_obstacle_acceleration()`: 计算障碍物加速度
  - `calculate_danger_level()`: 计算危险程度
  - `calculate_environmental_acceleration()`: 计算环境总加速度

### BoundaryManager
- **职责**：处理边界力
- **方法**：
  - `calculate_boundary_force()`: 计算边界力

## 使用示例

### 旧方式（不推荐）
```python
agent = Agent(pos, map_manager=map_manager)
agent.update_state(target, neighbors, obstacles, dangers, bound)
```

### 新方式（推荐）
```python
# 创建数据对象
agent = AgentData(pos, agent_type="adult")

# 创建管理器
agent_manager = AgentManager(map_manager)

# 使用管理器更新状态
agent_manager.update_state(agent, neighbors, obstacles, dangers, boundary_area)
```

## 优势

1. **清晰的状态管理**：所有状态更新都通过管理器进行，逻辑集中
2. **易于维护**：修改业务逻辑只需修改管理器类
3. **易于测试**：可以单独测试管理器类的方法
4. **易于扩展**：添加新功能只需添加新的管理器或扩展现有管理器
5. **数据与逻辑分离**：数据类可以轻松序列化/反序列化

## 迁移指南

### 从旧代码迁移

1. 将 `Agent(...)` 改为 `AgentData(...)`
2. 将 `Obs(...)` 改为 `ObstacleData(...)`
3. 将 `Dan(...)` 改为 `DangerData(...)`
4. 将 `Area(...)` 改为 `AreaData(...)`
5. 创建 `AgentManager` 并使用其 `update_state()` 方法
6. 使用管理器类的方法替代原来的类方法

### 向后兼容

为了保持向后兼容，我们保留了旧类名作为别名：
- `Agent` → `AgentData`
- `Obs` → `ObstacleData`
- `Dan` → `DangerData`
- `Area` → `AreaData`

但建议新代码使用新的数据类和管理器架构。
