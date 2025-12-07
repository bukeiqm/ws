# 监视器系统使用说明

## 概述

监视器系统可以实时监控特定行人的详细状态，包括各项加速度分量、路径选择权重和恐慌值等信息。这对于分析行人行为、调试参数和进行科学研究非常有用。

## 功能特性

1. **实时监控**：在模拟过程中实时显示被监视行人的状态
2. **加速度分解**：记录并显示各项加速度分量
3. **路径选择权重**：显示影响路径选择的各项因子权重
4. **恐慌值监控**：实时显示恐慌值变化
5. **独立窗口**：在单独的窗口中显示监视信息

## 使用方法

### 基本使用

```python
from pedestrian_evacuation import run_simulation

# 监视第一个行人（索引0）
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    monitor_agent_index=0  # 监视第一个行人
)
```

### 监视特定行人

```python
# 监视第5个行人（索引从0开始，所以是索引4）
ani = run_simulation(
    num_agents=20,
    monitor_agent_index=4
)
```

### 监视 + 火灾系统

```python
ani = run_simulation(
    num_agents=20,
    enable_fire=True,
    fire_initial_count=2,
    monitor_agent_index=0  # 观察火灾对行人的影响
)
```

### 监视 + AI路径推荐

```python
ani = run_simulation(
    num_agents=20,
    enable_ai=True,
    ai_recommendation_weight=0.5,
    monitor_agent_index=0  # 观察AI推荐的影响
)
```

## 监视器显示内容

### 基本信息
- **时间**：当前模拟时间
- **位置**：行人的当前位置 (x, y)
- **速度**：行人的当前速度大小
- **恐慌值**：当前的恐慌值（0-1）
- **已疏散**：是否已疏散

### 加速度分量 (m/s²)

监视器会显示以下加速度分量的大小（模长）：

1. **目标驱动** (`acc_goal`)
   - 由目标位置产生的加速度
   - 驱动行人朝向目标移动

2. **行人间作用** (`acc_int`)
   - 来自其他行人的相互作用加速度
   - 包括排斥力和从众力

3. **障碍物** (`acc_obstacle`)
   - 来自障碍物的环境力加速度
   - 推离障碍物

4. **普通危险源** (`acc_danger`)
   - 来自普通危险源的环境力加速度
   - 不包括火灾

5. **火灾** (`acc_fire`)
   - 来自火灾的环境力加速度
   - 火灾对行人的推力

6. **边界** (`acc_bound`)
   - 来自区域边界的力加速度
   - 防止行人越界

7. **心理因素** (`acc_psy`)
   - 由恐慌状态产生的加速度
   - 受恐慌值和概率因子影响

8. **总加速度** (`acc_total`)
   - 所有加速度分量的矢量和

### 路径选择权重

显示影响路径选择的各项因子权重：

1. **距离权重**：距离出口的权重
2. **危险权重**：危险程度的权重
3. **障碍物权重**：障碍物影响的权重
4. **从众权重**：从众行为的权重
5. **AI推荐权重**：AI推荐的权重（如果启用）

## 监视器窗口

当启用监视器时，会创建一个独立的窗口显示被监视行人的实时状态。窗口内容包括：

- 实时更新的状态信息
- 格式化的数据显示
- 清晰的分类显示

## 数据记录

监视器会持续记录以下数据：

- **时间序列**：每个时间点的记录
- **恐慌值历史**：恐慌值随时间的变化
- **加速度历史**：各项加速度的历史记录
- **路径权重历史**：路径选择权重的历史记录

这些数据可以用于后续分析。

## 使用场景

### 1. 参数调试

通过监视器观察不同参数对行人行为的影响：

```python
ani = run_simulation(
    num_agents=20,
    monitor_agent_index=0,
    enable_fire=True,
    fire_initial_count=1
)
# 观察火灾对加速度和恐慌值的影响
```

### 2. AI效果验证

验证AI路径推荐的效果：

```python
ani = run_simulation(
    num_agents=20,
    enable_ai=True,
    ai_recommendation_weight=0.5,
    monitor_agent_index=0
)
# 观察AI推荐权重和路径选择的变化
```

### 3. 行为分析

分析行人在不同情况下的行为：

```python
ani = run_simulation(
    num_agents=20,
    enable_fire=True,
    enable_ai=True,
    monitor_agent_index=0
)
# 综合分析火灾、AI等因素对行人的影响
```

## 注意事项

1. **索引范围**：`monitor_agent_index` 必须在有效范围内（0 到 num_agents-1）
2. **性能影响**：监视器会增加少量计算开销，但影响很小
3. **窗口管理**：监视器窗口和主模拟窗口是独立的，可以分别关闭
4. **数据更新**：监视器数据每帧更新，确保实时性

## 代码结构

### AgentMonitor类

主要方法：
- `record()`: 记录当前状态
- `get_acceleration_magnitudes()`: 获取加速度大小
- `get_path_weights_dict()`: 获取路径权重字典

### 数据类

- `AccelerationRecord`: 存储加速度分量
- `PathSelectionWeights`: 存储路径选择权重

## 示例代码

完整示例请参考 `monitor_example.py` 文件。
