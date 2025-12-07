# 火灾系统使用说明

## 概述

火灾系统使用**元胞自动机（Cellular Automaton）**模型来模拟火灾的扩散和熄灭过程。火灾会在场景中随机出现，根据时间扩散到相邻区域，并在一定时间后自动熄灭。

## 功能特性

1. **随机生成**：火灾可以在场景中随机出现
2. **时间扩散**：火灾经过随机时间（2-5秒）向相邻区域扩散
3. **自动熄灭**：火灾经过一定时间（10-20秒）后自动熄灭
4. **元胞自动机管理**：使用元胞自动机模型管理火灾状态
5. **调试参数**：提供丰富的调试参数接口

## 使用方法

### 基本使用

```python
from pedestrian_evacuation import run_simulation

# 启用火灾系统
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    enable_fire=True  # 启用火灾系统
)
```

### 自定义参数

```python
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    enable_fire=True,
    # 火灾调试参数
    fire_initial_count=2,              # 初始火灾数量
    fire_spawn_probability=0.001,      # 每帧生成火灾的概率
    fire_spread_time_min=2.0,          # 火灾扩散最小时间（秒）
    fire_spread_time_max=5.0,          # 火灾扩散最大时间（秒）
    fire_duration_min=10.0,            # 火灾持续最小时间（秒）
    fire_duration_max=20.0,            # 火灾持续最大时间（秒）
    fire_spread_probability=0.6         # 火灾扩散到相邻区域的概率
)
```

## 参数说明

### 火灾扩散参数

- **`fire_spread_time_min`** (默认: 2.0秒)
  - 火灾向相邻区域扩散的最小时间
  - 值越小，火灾扩散越快

- **`fire_spread_time_max`** (默认: 5.0秒)
  - 火灾向相邻区域扩散的最大时间
  - 值越大，火灾扩散越慢

### 火灾持续时间参数

- **`fire_duration_min`** (默认: 10.0秒)
  - 火灾持续的最小时间
  - 值越小，火灾熄灭越快

- **`fire_duration_max`** (默认: 20.0秒)
  - 火灾持续的最大时间
  - 值越大，火灾持续时间越长

### 火灾生成参数

- **`fire_initial_count`** (默认: 0)
  - 模拟开始时的初始火灾数量
  - 设置为0表示不自动生成初始火灾

- **`fire_spawn_probability`** (默认: 0.0)
  - 每帧生成新火灾的概率（0-1）
  - 设置为0表示不自动生成新火灾
  - 建议值：0.0001 - 0.01（根据需要的火灾频率调整）

### 火灾扩散概率

- **`fire_spread_probability`** (默认: 0.6)
  - 火灾扩散到相邻区域的概率（0-1）
  - 值越大，火灾越容易扩散
  - 1.0表示100%扩散，0.0表示不扩散

## 火灾对行人的影响

火灾会对行人产生以下影响：

1. **危险源影响**：火灾作为危险源，会影响行人的路径选择
   - 危险影响范围：`FIRE_DANGER_LAMBDA = 8.0`
   - 基础刺激值：`FIRE_DANGER_STIMULI = 2.0`（比普通危险源更高）

2. **环境力影响**：火灾产生环境力，推离行人
   - 环境力参数A：`FIRE_ENVIRONMENTAL_FORCE_A = 8.0`
   - 环境力参数B：`FIRE_ENVIRONMENTAL_FORCE_B = 1.0`

3. **恐慌值影响**：火灾会增加行人的恐慌值，影响其行为

## 可视化

火灾在可视化中显示为：
- **红色星形标记**：表示火灾位置（区域中心）
- **红色半透明矩形**：表示着火的区域
- **统计信息**：显示当前火灾数量

## 元胞自动机模型

火灾系统使用元胞自动机模型，每个区域（元胞）可以处于以下状态：
- **未着火**：正常状态
- **着火**：该区域有火灾

### 状态转换规则

1. **生成**：火灾可以在可通行区域随机生成
2. **扩散**：着火的区域经过随机时间（2-5秒）后，以一定概率（60%）扩散到相邻的可通行区域
3. **熄灭**：着火的区域经过随机时间（10-20秒）后自动熄灭

### 扩散规则

- 火灾只能扩散到**相邻的可通行区域**
- 不能扩散到**不可通行区域**（如货架）
- 不能扩散到**已有火灾的区域**
- 扩散概率由 `fire_spread_probability` 控制

## 调试建议

### 快速测试火灾扩散

```python
ani = run_simulation(
    enable_fire=True,
    fire_initial_count=1,
    fire_spread_time_min=1.0,  # 快速扩散
    fire_spread_time_max=2.0,
    fire_duration_min=15.0,    # 持续时间较长，便于观察
    fire_duration_max=25.0,
    fire_spread_probability=0.8  # 高扩散概率
)
```

### 缓慢扩散测试

```python
ani = run_simulation(
    enable_fire=True,
    fire_initial_count=1,
    fire_spread_time_min=5.0,   # 缓慢扩散
    fire_spread_time_max=8.0,
    fire_duration_min=8.0,
    fire_duration_max=12.0,
    fire_spread_probability=0.3  # 低扩散概率
)
```

### 多火灾测试

```python
ani = run_simulation(
    enable_fire=True,
    fire_initial_count=3,       # 多个初始火灾
    fire_spawn_probability=0.002,  # 持续生成新火灾
    fire_spread_probability=0.5
)
```

## 代码结构

### FireData（火灾数据类）
- 存储火灾的状态信息
- 位置：`area_id`（区域ID）
- 时间：`start_time`（开始时间）、`duration`（持续时间）
- 扩散：`spread_time`（扩散时间）、`next_spread_time`（下次扩散时间）

### FireManager（火灾管理器）
- 管理所有火灾的状态
- 实现元胞自动机的状态转换
- 提供火灾查询和可视化接口

### 主要方法

- `add_fire(area_id, current_time)`: 在指定区域添加火灾
- `remove_fire(area_id)`: 移除指定区域的火灾
- `update(current_time)`: 更新火灾状态（扩散和熄灭）
- `get_fire_dangers()`: 获取所有火灾作为危险源
- `is_area_on_fire(area_id)`: 检查区域是否着火
- `get_fire_count()`: 获取当前火灾数量

## 注意事项

1. 火灾只能在**可通行区域**生成和扩散
2. 火灾不会扩散到**出口区域**（避免阻塞逃生路径）
3. 火灾参数可以在运行时通过函数参数调整，也可以通过修改 `constants.py` 中的默认值
4. 火灾系统会增加计算开销，如果性能有问题，可以减少火灾数量或关闭自动生成

## 示例代码

完整示例请参考 `fire_example.py` 文件。
