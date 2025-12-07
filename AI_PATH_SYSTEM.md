# AI路径推荐系统使用说明

## 概述

AI路径推荐系统使用**MPC（模型预测控制）**算法为行人提供最优路径推荐。系统会预测未来路径上的人流密度和危险程度（如火灾），推荐人流更少、危险更少的路线。

## 功能特性

1. **基于MPC的路径规划**：使用模型预测控制算法预测未来状态
2. **人流密度预测**：预测路径上未来的人流密度
3. **危险程度预测**：预测路径上未来的危险程度（包括火灾）
4. **动态更新**：根据更新间隔定期更新推荐
5. **可调权重**：可以调整AI推荐的权重

## 使用方法

### 基本使用

```python
from pedestrian_evacuation import run_simulation

# 启用AI路径推荐
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    enable_ai=True  # 启用AI路径推荐
)
```

### 自定义参数

```python
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    enable_ai=True,
    # AI推荐参数
    ai_recommendation_weight=0.4,  # AI推荐权重（0-1）
    mpc_horizon=5,  # MPC预测时域（预测未来N步）
    mpc_update_interval=1.0  # AI推荐更新间隔（秒）
)
```

### AI + 火灾系统

```python
ani = run_simulation(
    use_smart_choice=True,
    num_agents=20,
    enable_fire=True,  # 启用火灾系统
    fire_initial_count=2,
    enable_ai=True,  # 启用AI路径推荐
    ai_recommendation_weight=0.5,  # AI会考虑火灾危险
    mpc_horizon=7  # 更长的预测时域
)
```

## 参数说明

### AI推荐权重

- **`ai_recommendation_weight`** (默认: 0.3)
  - AI推荐在路径选择中的权重（0-1）
  - 值越大，行人越信任AI推荐
  - 0.0：完全忽略AI推荐
  - 1.0：完全遵循AI推荐
  - 建议值：0.2-0.6

### MPC预测时域

- **`mpc_horizon`** (默认: 5)
  - MPC预测未来多少步
  - 值越大，预测越远，但计算开销也越大
  - 建议值：3-8

### AI更新间隔

- **`mpc_update_interval`** (默认: 1.0秒)
  - AI推荐更新的时间间隔
  - 值越小，更新越频繁，但计算开销也越大
  - 建议值：0.5-2.0秒

## MPC算法原理

### 代价函数

MPC使用以下代价函数评估路径：

```
代价 = 距离代价 + 密度代价 + 危险代价
```

其中：
- **距离代价**：路径到出口的距离
- **密度代价**：预测路径上的人流密度
- **危险代价**：预测路径上的危险程度（包括火灾）

### 预测机制

1. **人流密度预测**：
   - 预测未来一段时间内路径上的人流密度
   - 考虑所有行人的当前位置和速度

2. **危险程度预测**：
   - 预测路径上的危险源（包括火灾）
   - 考虑火灾的扩散可能性
   - 相邻区域有火灾时，危险值增加

3. **路径优化**：
   - 计算所有可选路径的代价
   - 选择代价最小的路径作为推荐

## AI推荐在路径选择中的作用

AI推荐作为路径选择的一个因素，与其他因素（距离、危险、障碍物、从众行为）一起影响最终选择：

```
总得分 = 距离得分 × 距离权重 +
        危险得分 × 危险权重 +
        障碍物得分 × 障碍物权重 +
        从众得分 × 从众权重 +
        AI推荐得分 × AI权重
```

AI推荐的得分：
- 如果路径是AI推荐的：得分 = 2.0
- 如果路径不是AI推荐的：得分 = 0.5

## 使用建议

### 高AI权重场景

适用于：
- 需要快速疏散的场景
- 火灾等危险情况
- 需要优化整体疏散效率

```python
ani = run_simulation(
    enable_ai=True,
    ai_recommendation_weight=0.6,  # 高权重
    mpc_horizon=7,  # 长预测时域
    mpc_update_interval=0.5  # 频繁更新
)
```

### 低AI权重场景

适用于：
- AI仅作为参考
- 保留更多行人自主决策
- 测试AI效果

```python
ani = run_simulation(
    enable_ai=True,
    ai_recommendation_weight=0.2,  # 低权重
    mpc_horizon=4,
    mpc_update_interval=2.0
)
```

### 平衡场景

```python
ani = run_simulation(
    enable_ai=True,
    ai_recommendation_weight=0.4,  # 平衡权重
    mpc_horizon=5,
    mpc_update_interval=1.0
)
```

## 性能考虑

1. **计算开销**：
   - MPC算法需要计算所有可选路径的代价
   - 预测时域越大，计算开销越大
   - 更新间隔越小，计算开销越大

2. **优化建议**：
   - 对于大量行人，适当增大更新间隔
   - 对于实时性要求不高的场景，可以增大预测时域
   - 可以通过调整权重来平衡AI影响和计算开销

## 与火灾系统的协同

AI路径推荐系统会考虑火灾：
1. **火灾区域**：直接避开着火区域
2. **相邻火灾**：考虑火灾可能扩散到相邻区域
3. **火灾危险**：在代价函数中给予火灾高权重

```python
ani = run_simulation(
    enable_fire=True,
    fire_initial_count=2,
    enable_ai=True,
    ai_recommendation_weight=0.5,  # AI会主动避开火灾
    mpc_horizon=6  # 预测火灾扩散
)
```

## 代码结构

### MPCPlanner类

主要方法：
- `recommend_path()`: 推荐最优路径
- `predict_path_density()`: 预测路径人流密度
- `predict_path_danger()`: 预测路径危险程度
- `compute_path_cost()`: 计算路径代价

### 集成点

AI推荐集成在 `PathPlanner.choose_path()` 方法中：
- 如果启用AI，获取AI推荐
- 将AI推荐作为得分项加入路径选择
- 根据权重调整AI影响

## 注意事项

1. **必须启用智能路径选择**：AI推荐只在 `use_smart_choice=True` 时生效
2. **恐慌时忽略AI**：当行人恐慌值超过阈值时，会随机选择，忽略AI推荐
3. **缓存机制**：AI推荐会缓存，根据更新间隔更新，减少计算开销
4. **路径可达性**：AI只推荐可达出口的路径

## 示例代码

完整示例请参考 `ai_path_example.py` 文件。
