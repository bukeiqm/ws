"""行人疏散模拟系统 - 模拟和可视化模块"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from . import constants
from .models import Map
from .entities import AgentData, ObstacleData, DangerData, AreaData, FireData
from .managers import AgentManager, FireManager, MPCPlanner
from .monitor import AgentMonitor

# 设置字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 模拟参数
NUM_AGENTS = 10
NUM_OBSTACLES = 10
NUM_DANGER = 10


def get_random_position_in_passable_area(map_manager, passable_areas, margin=0.3):
    """在可通行区域内随机生成位置"""
    if len(passable_areas) == 0:
        return None
    
    # 随机选择一个可通行区域
    area_id = np.random.choice(passable_areas)
    area = map_manager.areas[area_id]
    
    # 在区域内随机位置（避开边界margin距离）
    lb = area.left_bottom
    rt = area.right_top
    pos = np.array([
        np.random.uniform(lb[0] + margin, rt[0] - margin),
        np.random.uniform(lb[1] + margin, rt[1] - margin)
    ])
    
    return pos


def plot_agent_parameters(agent, figsize=(16, 12)):
    """绘制某个行人的各个参数随时间的变化
    
    Args:
        agent: Agent对象，需要已启用数据记录（agent.record_data=True）
        figsize: 图形大小
    """
    if not agent.record_data or len(agent.data_history['time']) == 0:
        print("警告：该行人未启用数据记录或没有数据")
        return None
    
    time = agent.data_history['time']
    
    # 创建子图
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle('行人参数随时间变化', fontsize=16, fontweight='bold')
    
    # 1. 速度
    axes[0, 0].plot(time, agent.data_history['speed'], 'b-', linewidth=2)
    axes[0, 0].set_title('速度 (m/s)', fontsize=12)
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('速度 (m/s)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 各种加速度
    axes[0, 1].plot(time, agent.data_history['acc_goal_magnitude'], 'g-', label='目标加速度', linewidth=2)
    axes[0, 1].plot(time, agent.data_history['acc_int_magnitude'], 'r-', label='群体加速度', linewidth=2)
    axes[0, 1].plot(time, agent.data_history['acc_env_magnitude'], 'orange', label='环境加速度', linewidth=2)
    axes[0, 1].plot(time, agent.data_history['acc_bound_magnitude'], 'purple', label='边界加速度', linewidth=2)
    axes[0, 1].plot(time, agent.data_history['acc_psy_magnitude'], 'brown', label='心理加速度', linewidth=2)
    axes[0, 1].set_title('加速度分量 (m/s²)', fontsize=12)
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('加速度大小 (m/s²)')
    axes[0, 1].legend(loc='best', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 感知范围内人数
    axes[1, 0].plot(time, agent.data_history['neighbor_count'], 'c-', linewidth=2)
    axes[1, 0].set_title('感知范围内人数', fontsize=12)
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('人数')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 路径选择得分
    axes[1, 1].plot(time, agent.data_history['path_distance_score'], 'g-', label='距离得分', linewidth=2)
    axes[1, 1].plot(time, agent.data_history['path_danger_score'], 'r-', label='危险得分', linewidth=2)
    axes[1, 1].plot(time, agent.data_history['path_obstacle_score'], 'orange', label='障碍物得分', linewidth=2)
    axes[1, 1].plot(time, agent.data_history['path_herd_score'], 'purple', label='从众得分', linewidth=2)
    axes[1, 1].set_title('路径选择得分', fontsize=12)
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('得分')
    axes[1, 1].legend(loc='best', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    max_herd = max(agent.data_history['path_herd_score']) if agent.data_history['path_herd_score'] else 1.0
    axes[1, 1].set_ylim([0, max(2.0, max_herd)])
    
    # 5. 加速度对比（堆叠图）
    acc_goal = agent.data_history['acc_goal_magnitude']
    acc_int = agent.data_history['acc_int_magnitude']
    acc_env = agent.data_history['acc_env_magnitude']
    axes[2, 0].fill_between(time, 0, acc_goal, alpha=0.6, label='目标', color='green')
    axes[2, 0].fill_between(time, acc_goal, 
                           [a+b for a, b in zip(acc_goal, acc_int)], 
                           alpha=0.6, label='群体', color='red')
    axes[2, 0].fill_between(time, 
                           [a+b for a, b in zip(acc_goal, acc_int)],
                           [a+b+c for a, b, c in zip(acc_goal, acc_int, acc_env)],
                           alpha=0.6, label='环境', color='orange')
    axes[2, 0].set_title('加速度分量堆叠图', fontsize=12)
    axes[2, 0].set_xlabel('时间 (s)')
    axes[2, 0].set_ylabel('加速度大小 (m/s²)')
    axes[2, 0].legend(loc='best', fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. 总加速度
    total_acc = [a+b+c+d+e for a, b, c, d, e in zip(
        agent.data_history['acc_goal_magnitude'],
        agent.data_history['acc_int_magnitude'],
        agent.data_history['acc_env_magnitude'],
        agent.data_history['acc_bound_magnitude'],
        agent.data_history['acc_psy_magnitude']
    )]
    axes[2, 1].plot(time, total_acc, 'k-', linewidth=2)
    axes[2, 1].set_title('总加速度大小', fontsize=12)
    axes[2, 1].set_xlabel('时间 (s)')
    axes[2, 1].set_ylabel('总加速度 (m/s²)')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 7. 速度与总加速度对比
    ax3_twin = axes[3, 0].twinx()
    line1 = axes[3, 0].plot(time, agent.data_history['speed'], 'b-', linewidth=2, label='速度')
    line2 = ax3_twin.plot(time, total_acc, 'r-', linewidth=2, label='总加速度')
    axes[3, 0].set_xlabel('时间 (s)')
    axes[3, 0].set_ylabel('速度 (m/s)', color='b')
    ax3_twin.set_ylabel('总加速度 (m/s²)', color='r')
    axes[3, 0].tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    axes[3, 0].set_title('速度与总加速度对比', fontsize=12)
    axes[3, 0].grid(True, alpha=0.3)
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[3, 0].legend(lines, labels, loc='best')
    
    # 8. 路径选择得分堆叠
    dist_score = agent.data_history['path_distance_score']
    danger_score = agent.data_history['path_danger_score']
    obs_score = agent.data_history['path_obstacle_score']
    axes[3, 1].fill_between(time, 0, dist_score, alpha=0.6, label='距离', color='green')
    axes[3, 1].fill_between(time, dist_score,
                           [a+b for a, b in zip(dist_score, danger_score)],
                           alpha=0.6, label='危险', color='red')
    axes[3, 1].fill_between(time,
                           [a+b for a, b in zip(dist_score, danger_score)],
                           [a+b+c for a, b, c in zip(dist_score, danger_score, obs_score)],
                           alpha=0.6, label='障碍物', color='orange')
    axes[3, 1].set_title('路径选择得分堆叠图', fontsize=12)
    axes[3, 1].set_xlabel('时间 (s)')
    axes[3, 1].set_ylabel('得分')
    axes[3, 1].legend(loc='best', fontsize=9)
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_simulation(use_smart_choice=None, num_agents=None, num_obstacles=None, num_danger=None,
                  enable_fire=False, fire_initial_count=None, fire_spawn_probability=None,
                  fire_spread_time_min=None, fire_spread_time_max=None,
                  fire_duration_min=None, fire_duration_max=None,
                  fire_spread_probability=None,
                  enable_ai=False, ai_recommendation_weight=None,
                  mpc_horizon=None, mpc_update_interval=None,
                  monitor_agent_index=None):
    """运行仿真
    
    Args:
        use_smart_choice: 是否使用智能路径选择（None时使用全局变量）
        num_agents: 行人数量（None时使用默认值）
        num_obstacles: 障碍物数量（None时使用默认值）
        num_danger: 危险源数量（None时使用默认值）
        enable_fire: 是否启用火灾系统
        fire_initial_count: 初始火灾数量（None时使用constants中的值）
        fire_spawn_probability: 每帧生成火灾的概率（None时使用constants中的值）
        fire_spread_time_min: 火灾扩散最小时间（None时使用constants中的值）
        fire_spread_time_max: 火灾扩散最大时间（None时使用constants中的值）
        fire_duration_min: 火灾持续最小时间（None时使用constants中的值）
        fire_duration_max: 火灾持续最大时间（None时使用constants中的值）
        fire_spread_probability: 火灾扩散概率（None时使用constants中的值）
        enable_ai: 是否启用AI路径推荐（MPC）
        ai_recommendation_weight: AI推荐权重（0-1，None时使用constants中的值）
        mpc_horizon: MPC预测时域（None时使用constants中的值）
        mpc_update_interval: AI推荐更新间隔（秒，None时使用constants中的值）
        monitor_agent_index: 要监视的行人索引（None表示不监视，0表示第一个行人）
    """
    # 使用传入参数，默认使用智能选择
    if use_smart_choice is None:
        use_smart_choice = True
    
    # 使用传入参数或默认值
    num_agents = num_agents if num_agents is not None else NUM_AGENTS
    num_obstacles = num_obstacles if num_obstacles is not None else NUM_OBSTACLES
    num_danger = num_danger if num_danger is not None else NUM_DANGER
    
    # 创建地图管理器
    map_manager = Map()
    
    # ========== 根据购物中心平面图创建地图 ==========
    # 地图尺寸：根据图片布局设计
    map_width = 24.0  # 地图宽度（米）
    map_height = 18.0  # 地图高度（米）
    cell_size = 2.0  # 每个区块的大小（米）
    
    # 创建网格系统（12x9 网格）
    grid_cols = 12
    grid_rows = 9
    
    # 定义购物中心布局
    # 2=路口，1=道路，0=不可通行/货架，3=出口（同时也是路口）
    # 注意：如果layout中只有1和0，系统会自动将1视为路口（向后兼容）
    layout = [
        # 行0（顶部）
        [1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1],  # 顶部通道（包含顶部出口）
        # 行1
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],  # 肉区、火锅食材区、面点区等
        # 行2
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],  # 货架区域
        # 行3
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 生鲜区通道（右侧出口）
        # 行4
        [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 3],  # 生鲜区、食品区（右侧出口）
        # 行5
        [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],  # 货架区域
        # 行6
        [1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1],  # 中部通道（包含底部出口）
        # 行7
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 底部货架区域
        # 行8（底部）
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 底部通道
    ]
    
    # 检查layout中是否有值为2或3的区域（路口或出口）
    # 如果没有，则向后兼容：将1视为路口
    has_nodes = any(2 in row or 3 in row for row in layout)
    
    # 创建区块
    # 根据layout数组：2=路口，1=道路，0=不可通行区域，3=出口（同时也是路口）
    # 如果layout中没有2或3，则向后兼容：将1视为路口
    passable_areas = []
    area_grid = {}  # 存储每个网格位置对应的区域ID
    node_grid = {}  # 存储路口位置对应的区域ID（用于道路连接）
    exits = []  # 存储出口信息 [(area_id, exit_pos), ...]
    
    if has_nodes:
        # 使用新模式：2=路口，1=道路，0=不可通行，3=出口（同时也是路口）
        
        # 第一步：创建所有路口（值为2或3的区域）
        for j in range(grid_rows):
            for i in range(grid_cols):
                if layout[j][i] == 2 or layout[j][i] == 3:
                    lb = [i * cell_size, j * cell_size]
                    rt = [(i + 1) * cell_size, (j + 1) * cell_size]
                    area_id = map_manager.add_node(lb, rt)
                    passable_areas.append(area_id)
                    area_grid[(i, j)] = area_id
                    node_grid[(i, j)] = area_id
                    
                    # 如果值为3，标记为出口
                    if layout[j][i] == 3:
                        exit_area = map_manager.areas[area_id]
                        exit_pos = np.array(exit_area.center)
                        exits.append((area_id, exit_pos))
        
        # 第二步：创建所有道路（值为1的区域），并连接到相邻的路口
        for j in range(grid_rows):
            for i in range(grid_cols):
                if layout[j][i] == 1:
                    lb = [i * cell_size, j * cell_size]
                    rt = [(i + 1) * cell_size, (j + 1) * cell_size]
                    
                    # 查找相邻的路口（用于连接道路）
                    node1_id = None
                    node2_id = None
                    
                    # 检查四个方向的相邻路口
                    neighbors = [
                        (i+1, j),  # 右
                        (i-1, j),  # 左
                        (i, j+1),  # 上
                        (i, j-1)   # 下
                    ]
                    
                    adjacent_nodes = []
                    for ni, nj in neighbors:
                        if (ni, nj) in node_grid:
                            adjacent_nodes.append(node_grid[(ni, nj)])
                    
                    # 如果找到相邻路口，连接前两个（如果有的话）
                    if len(adjacent_nodes) >= 1:
                        node1_id = adjacent_nodes[0]
                    if len(adjacent_nodes) >= 2:
                        node2_id = adjacent_nodes[1]
                    
                    # 创建道路
                    area_id = map_manager.add_road(lb, rt, node1_id, node2_id)
                    passable_areas.append(area_id)
                    area_grid[(i, j)] = area_id
        
        # 第三步：创建不可通行区域（值为0的区域）
        for j in range(grid_rows):
            for i in range(grid_cols):
                if layout[j][i] == 0:
                    lb = [i * cell_size, j * cell_size]
                    rt = [(i + 1) * cell_size, (j + 1) * cell_size]
                    map_manager.add_forbidden_area(lb, rt)
        
        # 建立路口和道路之间的连接关系（相邻的可通行区域连接）
        for (i, j), area_id in area_grid.items():
            # 检查四个方向的相邻区块
            neighbors = [
                (i+1, j),  # 右
                (i-1, j),  # 左
                (i, j+1),  # 上
                (i, j-1)   # 下
            ]
            
            for ni, nj in neighbors:
                if (ni, nj) in area_grid:
                    neighbor_id = area_grid[(ni, nj)]
                    map_manager.connect_areas(area_id, neighbor_id)
    else:
        # 向后兼容模式：1=路口（可通行），0=不可通行，3=出口（同时也是路口）
        for j in range(grid_rows):
            for i in range(grid_cols):
                lb = [i * cell_size, j * cell_size]
                rt = [(i + 1) * cell_size, (j + 1) * cell_size]
                
                if layout[j][i] == 1 or layout[j][i] == 3:
                    # 可通行区块（视为路口）
                    area_id = map_manager.add_node(lb, rt)
                    passable_areas.append(area_id)
                    area_grid[(i, j)] = area_id
                    
                    # 如果值为3，标记为出口
                    if layout[j][i] == 3:
                        exit_area = map_manager.areas[area_id]
                        exit_pos = np.array(exit_area.center)
                        exits.append((area_id, exit_pos))
                else:
                    # 不可通行区块（货架）
                    map_manager.add_forbidden_area(lb, rt)
        
        # 建立区块之间的连接关系（相邻的可通行区块连接）
        for (i, j), area_id in area_grid.items():
            # 检查四个方向的相邻区块
            neighbors = [
                (i+1, j),  # 右
                (i-1, j),  # 左
                (i, j+1),  # 上
                (i, j-1)   # 下
            ]
            
            for ni, nj in neighbors:
                if (ni, nj) in area_grid:
                    neighbor_id = area_grid[(ni, nj)]
                    map_manager.connect_areas(area_id, neighbor_id)
        
        # 如果没有从layout中找到出口，使用默认位置
        if len(exits) == 0:
            if len(passable_areas) > 0:
                default_area_id = passable_areas[0]
                default_area = map_manager.areas[default_area_id]
                exits.append((default_area_id, np.array(default_area.center)))
        
        # 使用第一个出口作为主要出口
        if len(exits) > 0:
            exit_node_id, target_pos = exits[0]
        else:
            exit_node_id = -1
            target_pos = np.array([0, 0])
    
    # 如果没有从layout中找到出口，使用默认位置（向后兼容）
    if len(exits) == 0:
        if len(passable_areas) > 0:
            default_area_id = passable_areas[0]
            default_area = map_manager.areas[default_area_id]
            exits.append((default_area_id, np.array(default_area.center)))
    
    # 使用第一个出口作为主要出口（可以根据需要修改）
    if len(exits) > 0:
        exit_node_id, target_pos = exits[0]
    else:
        # 如果没有出口，使用默认值
        exit_node_id = -1
        target_pos = np.array([0, 0])

    # 创建Agent管理器
    agent_manager = AgentManager(map_manager)
    
    # 创建AI路径推荐器（如果启用）
    mpc_planner = None
    if enable_ai:
        # 应用AI参数（如果提供）
        constants.MPC_ENABLED = True
        if ai_recommendation_weight is not None:
            constants.MPC_AI_RECOMMENDATION_WEIGHT = ai_recommendation_weight
        if mpc_horizon is not None:
            constants.MPC_HORIZON = mpc_horizon
        if mpc_update_interval is not None:
            constants.MPC_UPDATE_INTERVAL = mpc_update_interval
        
        mpc_planner = MPCPlanner(map_manager)
    else:
        constants.MPC_ENABLED = False
    
    # 创建火灾管理器（如果启用）
    fire_manager = None
    if enable_fire:
        fire_manager = FireManager(map_manager)
        
        # 应用调试参数（如果提供）
        if fire_spread_time_min is not None:
            constants.FIRE_SPREAD_TIME_MIN = fire_spread_time_min
        if fire_spread_time_max is not None:
            constants.FIRE_SPREAD_TIME_MAX = fire_spread_time_max
        if fire_duration_min is not None:
            constants.FIRE_DURATION_MIN = fire_duration_min
        if fire_duration_max is not None:
            constants.FIRE_DURATION_MAX = fire_duration_max
        if fire_spread_probability is not None:
            constants.FIRE_SPREAD_PROBABILITY = fire_spread_probability
        if fire_spawn_probability is not None:
            constants.FIRE_SPAWN_PROBABILITY = fire_spawn_probability
        
        # 初始化火灾（如果设置了初始数量）
        initial_fire_count = fire_initial_count if fire_initial_count is not None else constants.FIRE_INITIAL_COUNT
        if initial_fire_count > 0 and len(passable_areas) > 0:
            for _ in range(initial_fire_count):
                # 随机选择一个可通行区域（排除出口区域）
                available_areas = [aid for aid in passable_areas if aid != exit_node_id]
                if available_areas:
                    fire_area_id = np.random.choice(available_areas)
                    fire_manager.add_fire(fire_area_id, current_time=0.0)
    
    # Set agents（在可通行区域内随机分布）
    agents = []
    for _ in range(num_agents):
        # 随机选择一个可通行区域
        if len(passable_areas) > 0:
            start_area_id = np.random.choice(passable_areas)
            start_area = map_manager.areas[start_area_id]
            # 在区域内随机位置
            lb = start_area.left_bottom
            rt = start_area.right_top
            pos = np.array([
                np.random.uniform(lb[0] + 0.3, rt[0] - 0.3),
                np.random.uniform(lb[1] + 0.3, rt[1] - 0.3)
            ])
        else:
            pos = np.random.uniform(0, 2, 2)
        
        agent = AgentData(pos, agent_type="adult")
        # 随机选择一个出口作为目标
        exit_idx = np.random.randint(0, len(exits))
        agent.exit_node_id, agent.exit_pos = exits[exit_idx]
        agent.exit_pos = np.array(agent.exit_pos, dtype=float)
        agent.use_smart_choice = use_smart_choice  # 设置路径选择方法
        agents.append(agent)

    # 创建监视器（如果指定了要监视的行人）
    monitor = None
    if monitor_agent_index is not None and 0 <= monitor_agent_index < len(agents):
        monitor = AgentMonitor(agents[monitor_agent_index])
        print(f"监视器已创建，监视行人 #{monitor_agent_index}")

    # Set obstacles（在可通行区域内随机生成）
    obstacles = []
    for _ in range(num_obstacles):
        pos = get_random_position_in_passable_area(map_manager, passable_areas, margin=0.3)
        if pos is not None:
            obs = ObstacleData(pos)
            obstacles.append(obs)

    # Set dangers（在可通行区域内随机生成）
    dangers = []
    for _ in range(num_danger):
        pos = get_random_position_in_passable_area(map_manager, passable_areas, margin=0.3)
        if pos is not None:
            dan = DangerData(pos)
            dangers.append(dan)

    # set bounds（使用地图中的不可通行区域作为边界）
    bounds = []
    for area_id, area in map_manager.areas.items():
        if not map_manager.is_area_passable(area_id):
            bounds.append(area)
    
    # 添加地图外边界
    map_bound = AreaData([-0.5, -0.5], [map_width+0.5, map_height+0.5], -1, 0)
    map_bound.jammed = ['left', 'right', 'upper', 'lower']
    bounds.append(map_bound)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1, map_width + 1)
    ax.set_ylim(-1, map_height + 1)
    ax.set_xlabel('X位置 (m)', fontsize=12)
    ax.set_ylabel('Y位置 (m)', fontsize=12)
    ax.set_title('购物中心安全疏散模拟 - 福利隆购物中心', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 绘制所有区块，并存储矩形引用以便动态更新
    area_rects = {}  # {area_id: Rectangle} 存储每个区域的矩形，用于动态更新颜色
    for area_id, area in map_manager.areas.items():
        lb = area.left_bottom
        rt = area.right_top
        width = rt[0] - lb[0]
        height = rt[1] - lb[1]
        
        if map_manager.is_area_passable(area_id):
            # 检查区域状态：着火、已熄灭或普通
            if fire_manager is not None:
                if fire_manager.is_area_on_fire(area_id):
                    # 着火区域：红色
                    face_color = 'red'
                    edge_color = 'darkred'
                    alpha = 0.3
                    linewidth = 2
                elif fire_manager.is_area_extinguished(area_id):
                    # 已熄灭区域：深灰色
                    face_color = 'dimgray'
                    edge_color = 'black'
                    alpha = 0.5
                    linewidth = 2
                else:
                    # 普通可通行区域：浅绿色
                    face_color = 'lightgreen'
                    edge_color = 'green'
                    alpha = 0.3
                    linewidth = 1
            else:
                # 未启用火灾系统：普通可通行区域
                face_color = 'lightgreen'
                edge_color = 'green'
                alpha = 0.3
                linewidth = 1
            
            rect = Rectangle(lb, width, height, 
                           facecolor=face_color, edgecolor=edge_color, 
                           linewidth=linewidth, alpha=alpha)
            ax.add_patch(rect)
            area_rects[area_id] = rect  # 存储引用以便后续更新
        else:
            # 不可通行区块：红色
            rect = Rectangle(lb, width, height, 
                           facecolor='red', edgecolor='darkred', 
                           linewidth=2, alpha=0.5)
            ax.add_patch(rect)
    
    # 绘制所有出口
    for exit_id, exit_pos in exits:
        ax.plot([exit_pos[0]], [exit_pos[1]], 'go', markersize=20, 
                markeredgewidth=3, markeredgecolor='green', label='安全出口', zorder=10)
    # 只显示一次图例
    if len(exits) > 0:
        ax.plot([], [], 'go', markersize=20, markeredgewidth=3, 
                markeredgecolor='green', label='安全出口')
    
    # 统计信息文本
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    scat1 = ax.scatter([p.pos[0] for p in agents if not p.evacuated],
                       [p.pos[1] for p in agents if not p.evacuated],
                       c='blue', marker='o', s=80, label='行人', zorder=5)
    scat2 = ax.scatter([o.pos[0] for o in obstacles],
                       [o.pos[1] for o in obstacles],
                       c='red', marker='s', s=150, label='障碍物', zorder=4)
    scat3 = ax.scatter([d.pos[0] for d in dangers],
                       [d.pos[1] for d in dangers],
                       c='orange', marker='^', s=200, label='危险源', zorder=4)
    
    # 火灾可视化（如果启用）
    scat4 = None
    fire_patches = {}  # {area_id: Rectangle} 存储火灾区域的矩形
    if fire_manager is not None:
        scat4 = ax.scatter([], [], c='red', marker='*', s=300, 
                          label='火灾', zorder=6, edgecolors='darkred', linewidths=2)
    
    evacuated_count = [0]  # 使用列表以便在闭包中修改
    
    # 创建监视器显示窗口（如果启用了监视器）
    monitor_fig = None
    monitor_ax = None
    monitor_text = None
    if monitor is not None:
        monitor_fig, monitor_ax = plt.subplots(figsize=(10, 8))
        monitor_ax.axis('off')
        monitor_ax.set_title(f'行人 #{monitor_agent_index} 实时监视器', fontsize=16, fontweight='bold', pad=20)
        monitor_text = monitor_ax.text(0.05, 0.95, '', transform=monitor_ax.transAxes,
                                       fontsize=11, verticalalignment='top',
                                       family='SimHei',
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        monitor_fig.tight_layout()
    
    def update(frame):
        current_time = frame * constants.SIMULATION_DT
        
        # 更新火灾系统
        fire_dangers = []
        if fire_manager is not None:
            # 更新火灾（扩散和熄灭）
            fire_manager.update(current_time)
            
            # 随机生成新火灾（如果启用）
            if constants.FIRE_SPAWN_PROBABILITY > 0 and len(passable_areas) > 0:
                if np.random.random() < constants.FIRE_SPAWN_PROBABILITY:
                    available_areas = [aid for aid in passable_areas 
                                     if aid != exit_node_id and not fire_manager.is_area_on_fire(aid)]
                    if available_areas:
                        fire_area_id = np.random.choice(available_areas)
                        fire_manager.add_fire(fire_area_id, current_time)
            
            # 获取火灾作为危险源
            fire_dangers = fire_manager.get_fire_dangers()
            
            # 更新火灾可视化
            fire_positions = fire_manager.get_fire_positions()
            if len(fire_positions) > 0:
                fire_array = np.array(fire_positions)
                scat4.set_offsets(fire_array)
            else:
                scat4.set_offsets(np.empty((0, 2)))
            
            # 更新火灾区域的矩形显示
            current_fire_areas = set(fire_manager.fire_areas.keys())
            # 移除已熄灭的火灾矩形
            for area_id in list(fire_patches.keys()):
                if area_id not in current_fire_areas:
                    fire_patches[area_id].remove()
                    del fire_patches[area_id]
            # 添加新火灾的矩形
            for area_id in current_fire_areas:
                if area_id not in fire_patches and area_id in map_manager.areas:
                    area = map_manager.areas[area_id]
                    lb = area.left_bottom
                    rt = area.right_top
                    width = rt[0] - lb[0]
                    height = rt[1] - lb[1]
                    fire_rect = Rectangle(lb, width, height, 
                                          facecolor='red', edgecolor='darkred',
                                          linewidth=3, alpha=0.4, zorder=3)
                    ax.add_patch(fire_rect)
                    fire_patches[area_id] = fire_rect
            
            # 更新已熄灭区域的背景颜色（深灰色/棕色，表示曾经着火）
            for area_id in fire_manager.extinguished_areas:
                if area_id in area_rects:
                    # 已熄灭区域：深灰色，表示曾经着火
                    area_rects[area_id].set_facecolor('dimgray')  # 深灰色
                    area_rects[area_id].set_edgecolor('black')
                    area_rects[area_id].set_alpha(0.5)
                    area_rects[area_id].set_linewidth(2)
            
            # 更新普通可通行区域（排除着火和已熄灭的区域）
            for area_id in passable_areas:
                if (area_id not in current_fire_areas and 
                    area_id not in fire_manager.extinguished_areas and
                    area_id in area_rects):
                    # 普通可通行区域：浅绿色
                    area_rects[area_id].set_facecolor('lightgreen')
                    area_rects[area_id].set_edgecolor('green')
                    area_rects[area_id].set_alpha(0.3)
                    area_rects[area_id].set_linewidth(1)
            
            # 更新当前着火区域的基础矩形（如果有的话，会被fire_patches覆盖，但保持一致性）
            for area_id in current_fire_areas:
                if area_id in area_rects:
                    # 着火区域的基础矩形也设为红色（虽然会被fire_patches覆盖）
                    area_rects[area_id].set_facecolor('red')
                    area_rects[area_id].set_edgecolor('darkred')
                    area_rects[area_id].set_alpha(0.3)
                    area_rects[area_id].set_linewidth(2)
        
        # 合并所有危险源（包括火灾）
        all_dangers = dangers + fire_dangers
        for i, agent in enumerate(agents):
            # 如果已疏散，跳过更新
            if agent.evacuated:
                continue
            
            # 查找当前所在区域
            current_area = map_manager.get_area_containing_point(agent.pos)
            if current_area:
                agent.current_area_id = current_area.id
                # 查找同一区域内的邻居
                neighbours = [a for a in agents if not a.evacuated and 
                            current_area.inside(a.pos)]
            else:
                agent.current_area_id = -1
                neighbours = []
            
            # 选择边界对象
            bound_obj = None
            if current_area and not map_manager.is_area_passable(current_area.id):
                bound_obj = current_area
            elif len(bounds) > 0:
                # 检查是否在任何边界内
                for b in bounds:
                    if b.inside(agent.pos):
                        bound_obj = b
                        break
                if bound_obj is None:
                    bound_obj = bounds[-1]  # 使用地图外边界
            
            # 使用AgentManager更新状态（包含火灾危险和AI推荐）
            agent_manager.update_state(
                agent, neighbours, obstacles, all_dangers, bound_obj,
                current_time=current_time,
                mpc_planner=mpc_planner,
                all_agents=agents,
                fire_manager=fire_manager
            )
            
            # 记录监视器数据（如果这是被监视的行人）
            if monitor is not None and agent is monitor.agent:
                if hasattr(agent, '_acc_record') and hasattr(agent, '_path_weights'):
                    from .monitor import PathSelectionWeights
                    path_weights = agent._path_weights if agent._path_weights else PathSelectionWeights()
                    monitor.record(current_time, agent._acc_record, path_weights)
        
        # 更新已疏散人数
        evacuated_count[0] = sum(1 for a in agents if a.evacuated)
        remaining_count = len(agents) - evacuated_count[0]
        
        # 更新散点图（只显示未疏散的行人）
        new_poses = np.array([p.pos for p in agents if not p.evacuated])
        if len(new_poses) > 0:
            scat1.set_offsets(new_poses)
        else:
            scat1.set_offsets(np.empty((0, 2)))
        
        # 更新统计信息
        stats_text_str = f'时间: {current_time:.1f}s\n'
        stats_text_str += f'已疏散: {evacuated_count[0]}/{len(agents)}\n'
        stats_text_str += f'剩余: {remaining_count}'
        if fire_manager is not None:
            stats_text_str += f'\n火灾: {fire_manager.get_fire_count()}处'
        stats_text.set_text(stats_text_str)
        
        # 更新监视器显示
        if monitor is not None and monitor_text is not None:
            monitor_str = f"=== 行人 #{monitor_agent_index} 实时状态 ===\n\n"
            monitor_str += f"时间: {current_time:.2f}s\n"
            monitor_str += f"位置: ({monitor.agent.pos[0]:.2f}, {monitor.agent.pos[1]:.2f})\n"
            monitor_str += f"速度: {np.linalg.norm(monitor.agent.vel):.2f} m/s\n"
            monitor_str += f"恐慌值: {monitor.current_panic:.3f}\n"
            monitor_str += f"已疏散: {'是' if monitor.agent.evacuated else '否'}\n\n"
            
            # 加速度信息
            if monitor.current_acceleration is not None:
                acc_mags = monitor.get_acceleration_magnitudes()
                monitor_str += "--- 加速度分量 (m/s²) ---\n"
                monitor_str += f"目标驱动:     {acc_mags['goal']:6.3f}\n"
                monitor_str += f"行人间作用:   {acc_mags['interpersonal']:6.3f}\n"
                monitor_str += f"障碍物:       {acc_mags['obstacle']:6.3f}\n"
                monitor_str += f"普通危险源:   {acc_mags['danger']:6.3f}\n"
                monitor_str += f"火灾:         {acc_mags['fire']:6.3f}\n"
                monitor_str += f"边界:         {acc_mags['boundary']:6.3f}\n"
                monitor_str += f"心理因素:     {acc_mags['psychological']:6.3f}\n"
                monitor_str += f"总加速度:     {acc_mags['total']:6.3f}\n\n"
            
            # 路径选择权重
            if monitor.current_path_weights is not None:
                weights = monitor.get_path_weights_dict()
                monitor_str += "--- 路径选择权重 ---\n"
                monitor_str += f"距离权重:     {weights['distance']:5.3f}\n"
                monitor_str += f"危险权重:     {weights['danger']:5.3f}\n"
                monitor_str += f"障碍物权重:   {weights['obstacle']:5.3f}\n"
                monitor_str += f"从众权重:     {weights['herd']:5.3f}\n"
                monitor_str += f"AI推荐权重:   {weights['ai']:5.3f}\n"
            
            monitor_text.set_text(monitor_str)
            monitor_fig.canvas.draw_idle()

        return_items = [scat1, stats_text]
        if scat4 is not None:
            return_items.append(scat4)
        return tuple(return_items)

    ani = FuncAnimation(fig, update, frames=300, interval=100, blit=False)
    
    # 如果启用了监视器，也更新监视器窗口
    if monitor is not None and monitor_fig is not None:
        def update_monitor(frame):
            # 监视器窗口通过update函数中的monitor_text更新
            pass
        monitor_ani = FuncAnimation(monitor_fig, update_monitor, frames=300, interval=100, blit=False)
        
    plt.legend()
    plt.show()  # 显示主仿真窗口

    return ani
