"""行人疏散模拟系统 - 模拟和可视化模块"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from . import constants
from .models import Map
from .entities import AgentData, ObstacleData, DangerData, AreaData, FireData
from .managers import AgentManager, FireManager, MPCPlanner, PathPlanner
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


def plot_agent_parameters(agent, figsize=(8, 6)):
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
    map_width = 28.0  # 地图宽度（米）
    map_height = 20.0  # 地图高度（米）
    cell_size = 2.0  # 每个区块的大小（米）
    
    # 创建网格系统（14x10 网格）
    grid_cols = 14
    grid_rows = 10
    
    # 定义购物中心布局
    # 0 = 不可通行/货架
    # 1 = 可通行区域
    # 3 = 出口（可通行，仅在地图边缘）
    # 根据福利隆购物中心平面图设计
    # 布局说明：
    # - 顶部：肉区、火锅食材区、面点区、楼梯、冷藏饮料
    # - 左侧：水产区米油区、楼梯、调料货架
    # - 中央：鲜蛋区、南北干调、杂粮区、冷冻区、生鲜区、散称区、熟食凉菜岛柜
    # - 底部：调料货架、方便面货架、酒货架、货架、客服区
    # - 右侧：饮料区、纸品区
    layout = [
        # 行0（顶部）- 顶部通道，包含顶部出口（在火锅食材区和面点区之间）
        [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],  # 顶部通道（肉区、火锅食材区、面点区、楼梯、冷藏饮料）
        # 行1 - 货架区域
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 肉区、火锅食材区、面点区、楼梯、冷藏饮料
        # 行2 - 货架区域
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # 货架区域
        # 行3 - 中央通道（生鲜区通道）
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # 生鲜区通道（鲜蛋区、南北干调、杂粮区、冷冻区、生鲜区、散称区）
        # 行4 - 中央区域
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],  # 生鲜区、食品区、散称区、熟食凉菜岛柜
        # 行5 - 中央区域
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # 货架区域
        # 行6 - 中部通道（包含底部出口，在方便面货架和酒货架之间）
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],  # 中部通道（调料货架、方便面货架、酒货架、货架、客服区）
        # 行7 - 底部货架区域
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],  # 底部货架区域（调料货架、方便面货架、酒货架、货架）
        # 行8 - 底部通道扩展
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # 底部通道扩展区域
        # 行9（底部）- 底部通道
        [0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],  # 底部通道
    ]
    
    layout = layout[::-1]

    # 创建区块
    passable_areas = []
    area_grid = {}  # 存储每个网格位置对应的区域ID
    exit_areas = []  # 存储出口区域信息 [(area_id, exit_pos), ...]
    
    for j in range(grid_rows):
        for i in range(grid_cols):
            lb = [i * cell_size, j * cell_size]
            rt = [(i + 1) * cell_size, (j + 1) * cell_size]
            
            if layout[j][i] == 1 or layout[j][i] == 3:
                # 可通行区块（包括出口）
                area_id = map_manager.add_node(lb, rt)
                passable_areas.append(area_id)
                area_grid[(i, j)] = area_id
                
                # 如果是出口（标记为3），检查是否在地图边缘
                if layout[j][i] == 3:
                    is_edge = (i == 0 or i == grid_cols - 1 or j == 0 or j == grid_rows - 1)
                    if is_edge:
                        # 计算出口位置（区域中心）
                        exit_pos = np.array([(lb[0] + rt[0]) / 2, (lb[1] + rt[1]) / 2])
                        exit_areas.append((area_id, exit_pos))
                    else:
                        # 如果出口不在边缘，发出警告并当作普通可通行区域处理
                        print(f"警告：位置 ({i}, {j}) 标记为出口但不在地图边缘，将作为普通可通行区域处理")
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
    
    # 从layout数组中读取出口位置
    exits = exit_areas.copy()
    
    # 如果没有找到出口，使用默认位置
    if len(exits) == 0:
        if len(passable_areas) > 0:
            default_area_id = passable_areas[0]
            default_area = map_manager.areas[default_area_id]
            exits.append((default_area_id, np.array(default_area.center)))
    
    # 使用第一个出口作为主要出口（可以根据需要修改）
    exit_node_id, target_pos = exits[0]

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
    
    # Set agents（在可通行区域内随机分布）
    agents = []
    # 定义行人类型列表
    agent_types = ["adult", "elder", "child"]
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
        
        # 随机选择行人类型（成年人、老人、儿童）
        agent_type = np.random.choice(agent_types)
        agent = AgentData(pos, agent_type=agent_type)
        agent.use_smart_choice = use_smart_choice  # 设置路径选择方法
        agents.append(agent)
    
    # 使用智能方法为每个行人选择最优出口
    path_planner = PathPlanner(map_manager)
    for agent in agents:
        # 获取当前位置的邻居（已创建的其他行人）
        current_area = map_manager.get_area_containing_point(agent.pos)
        neighbors = []
        if current_area:
            neighbors = [a for a in agents if a is not agent and 
                        current_area.inside(a.pos)]
        
        # 获取火灾危险源（如果启用）
        fire_dangers = []
        if fire_manager is not None:
            fire_dangers = fire_manager.get_fire_dangers()
        all_dangers = dangers + fire_dangers
        
        # 使用智能方法选择出口
        chosen_exit = path_planner.choose_exit(
            agent, exits, neighbors, all_dangers, obstacles,
            all_agents=agents, mpc_planner=mpc_planner, 
            fire_manager=fire_manager, current_time=0.0
        )
        
        if chosen_exit is not None:
            agent.exit_node_id, agent.exit_pos = chosen_exit
            agent.exit_pos = np.array(agent.exit_pos, dtype=float)
        else:
            # 如果选择失败，随机选择一个出口作为后备
            exit_idx = np.random.randint(0, len(exits))
            agent.exit_node_id, agent.exit_pos = exits[exit_idx]
            agent.exit_pos = np.array(agent.exit_pos, dtype=float)

    # 创建监视器（如果指定了要监视的行人）
    monitor = None
    if monitor_agent_index is not None and 0 <= monitor_agent_index < len(agents):
        monitor = AgentMonitor(agents[monitor_agent_index])
        print(f"监视器已创建，监视行人 #{monitor_agent_index}")

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
            # 不可通行区块：黑色（货架区域）
            rect = Rectangle(lb, width, height, 
                           facecolor='black', edgecolor='black', 
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
    # 为不同类型的行人设置不同颜色
    agent_type_colors = {
        "adult": 'blue',    # 成人：蓝色
        "elder": 'gray',    # 老人：灰色
        "child": 'cyan'     # 儿童：青色
    }
    
    # 按类型分组绘制行人
    scat_adult = ax.scatter([], [], c='blue', marker='o', s=80, label='成人', zorder=5)
    scat_elder = ax.scatter([], [], c='gray', marker='o', s=80, label='老人', zorder=5)
    scat_child = ax.scatter([], [], c='cyan', marker='o', s=80, label='儿童', zorder=5)
    scat_agents = {'adult': scat_adult, 'elder': scat_elder, 'child': scat_child}
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
    monitor_axes = None
    monitor_lines = {}  # 存储各个图表的线条对象
    if monitor is not None:
        # 创建包含多个子图的布局
        monitor_fig, monitor_axes = plt.subplots(2, 2, figsize=(8, 6))
        monitor_fig.suptitle(f'行人 #{monitor_agent_index} 实时监视器 - 数据随时间变化', 
                            fontsize=16, fontweight='bold')
        
        # 子图1：恐慌值随时间变化
        ax1 = monitor_axes[0, 0]
        ax1.set_title('恐慌值随时间变化', fontsize=12)
        ax1.set_xlabel('时间 (s)', fontsize=10)
        ax1.set_ylabel('恐慌值', fontsize=10)
        ax1.grid(True, alpha=0.3)
        # 不设置固定上限，让图表自动缩放以适应数据范围
        # 设置最小值为0，最大值由数据决定
        ax1.set_ylim(bottom=0)  # 只设置最小值，让最大值自动调整
        line_panic, = ax1.plot([], [], 'r-', linewidth=2, label='恐慌值')
        ax1.legend()
        monitor_lines['panic'] = line_panic
        
        # 子图2：各项加速度大小随时间变化
        ax2 = monitor_axes[0, 1]
        ax2.set_title('加速度分量随时间变化 (m/s²)', fontsize=12)
        ax2.set_xlabel('时间 (s)', fontsize=10)
        ax2.set_ylabel('加速度大小 (m/s²)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        lines_acc = {
            'goal': ax2.plot([], [], 'g-', linewidth=1.5, label='目标驱动', alpha=0.8)[0],
            'interpersonal': ax2.plot([], [], 'b-', linewidth=1.5, label='行人间作用', alpha=0.8)[0],
            'environmental': ax2.plot([], [], 'orange', linewidth=1.5, label='环境加速度', alpha=0.8)[0],
            'psychological': ax2.plot([], [], 'pink', linewidth=1.5, label='心理因素', alpha=0.8)[0],
            'total': ax2.plot([], [], 'k-', linewidth=2, label='总加速度', alpha=1.0)[0]
        }
        ax2.legend(loc='upper right', fontsize=9)
        monitor_lines['acceleration'] = lines_acc
        
        # 子图3：路径选择权重随时间变化
        ax3 = monitor_axes[1, 0]
        ax3.set_title('路径选择权重随时间变化', fontsize=12)
        ax3.set_xlabel('时间 (s)', fontsize=10)
        ax3.set_ylabel('权重', fontsize=10)
        ax3.grid(True, alpha=0.3)
        lines_weights = {
            'distance': ax3.plot([], [], 'g-', linewidth=1.5, label='距离权重', alpha=0.8)[0],
            'danger': ax3.plot([], [], 'r-', linewidth=1.5, label='危险权重', alpha=0.8)[0],
            'obstacle': ax3.plot([], [], 'orange', linewidth=1.5, label='障碍物权重', alpha=0.8)[0],
            'herd': ax3.plot([], [], 'purple', linewidth=1.5, label='从众权重', alpha=0.8)[0],
            'ai': ax3.plot([], [], 'blue', linewidth=1.5, label='AI推荐权重', alpha=0.8)[0]
        }
        ax3.legend(loc='upper right', fontsize=9)
        monitor_lines['weights'] = lines_weights
        
        # 子图4：速度随时间变化
        ax4 = monitor_axes[1, 1]
        ax4.set_title('速度随时间变化', fontsize=12)
        ax4.set_xlabel('时间 (s)', fontsize=10)
        ax4.set_ylabel('速度 (m/s)', fontsize=10)
        ax4.grid(True, alpha=0.3)
        line_speed, = ax4.plot([], [], 'b-', linewidth=2, label='速度')
        ax4.legend()
        monitor_lines['speed'] = line_speed
        
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
            if scat4 is not None:
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
                # 检查是否需要重新绘制（达到指定帧数后）
                if constants.MONITOR_RESET_FRAMES > 0:
                    # 初始化时间偏移（第一次记录时）
                    if monitor._frame_count == 0 and len(monitor.time_history) == 0:
                        monitor._time_offset = current_time
                    
                    monitor._frame_count += 1
                    
                    # 如果达到重置帧数，清除历史数据并重置时间基准
                    if monitor._frame_count >= constants.MONITOR_RESET_FRAMES:
                        monitor.clear_history()
                        monitor._frame_count = 0
                        # 重置时间基准，使时间从0开始
                        monitor._time_offset = current_time
                
                if hasattr(agent, '_acc_record') and hasattr(agent, '_path_weights'):
                    from .monitor import PathSelectionWeights
                    path_weights = agent._path_weights if agent._path_weights else PathSelectionWeights()
                    # 如果设置了时间偏移，调整记录的时间（使图表从0开始）
                    record_time = current_time
                    if constants.MONITOR_RESET_FRAMES > 0:
                        record_time = current_time - monitor._time_offset
                    monitor.record(record_time, agent._acc_record, path_weights)
        
        # 更新已疏散人数
        evacuated_count[0] = sum(1 for a in agents if a.evacuated)
        remaining_count = len(agents) - evacuated_count[0]
        
        # 更新散点图（只显示未疏散的行人，按类型分组）
        for agent_type, scat in scat_agents.items():
            type_poses = np.array([p.pos for p in agents if not p.evacuated and p.agent_type == agent_type])
            if len(type_poses) > 0:
                scat.set_offsets(type_poses)
            else:
                scat.set_offsets(np.empty((0, 2)))
        
        # 更新统计信息
        stats_text_str = f'时间: {current_time:.1f}s\n'
        stats_text_str += f'已疏散: {evacuated_count[0]}/{len(agents)}\n'
        stats_text_str += f'剩余: {remaining_count}'
        if fire_manager is not None:
            stats_text_str += f'\n火灾: {fire_manager.get_fire_count()}处'
        stats_text.set_text(stats_text_str)
        
        # 更新监视器显示（图表）
        if monitor is not None and monitor_axes is not None and monitor_lines and len(monitor.time_history) > 0:
            time_data = np.array(monitor.time_history)
            
            # 更新恐慌值图表
            if len(monitor.panic_history) > 0:
                monitor_lines['panic'].set_data(time_data, monitor.panic_history)
                monitor_axes[0, 0].relim()
                monitor_axes[0, 0].autoscale_view()
                # 确保y轴最小值从0开始，最大值根据数据自动调整
                y_min, y_max = monitor_axes[0, 0].get_ylim()
                monitor_axes[0, 0].set_ylim(bottom=0, top=max(y_max, max(monitor.panic_history) * 1.1))
            
            # 更新加速度图表
            if len(monitor.acceleration_history) > 0:
                acc_lines = monitor_lines['acceleration']
                # 计算环境加速度（障碍物 + 危险源 + 火灾 + 边界）
                acc_data = {
                    'goal': [float(np.linalg.norm(acc.acc_goal)) for acc in monitor.acceleration_history],
                    'interpersonal': [float(np.linalg.norm(acc.acc_int)) for acc in monitor.acceleration_history],
                    'environmental': [float(np.linalg.norm(acc.acc_obstacle + acc.acc_danger + 
                                                          acc.acc_fire + acc.acc_bound)) 
                                     for acc in monitor.acceleration_history],
                    'psychological': [float(np.linalg.norm(acc.acc_psy)) for acc in monitor.acceleration_history],
                    'total': [float(np.linalg.norm(acc.acc_total)) for acc in monitor.acceleration_history]
                }
                for key, line in acc_lines.items():
                    line.set_data(time_data, acc_data[key])
                monitor_axes[0, 1].relim()
                monitor_axes[0, 1].autoscale_view()
            
            # 更新路径选择权重图表
            if len(monitor.path_weights_history) > 0:
                weight_lines = monitor_lines['weights']
                weight_data = {
                    'distance': [w.distance_weight for w in monitor.path_weights_history],
                    'danger': [w.danger_weight for w in monitor.path_weights_history],
                    'obstacle': [w.obstacle_weight for w in monitor.path_weights_history],
                    'herd': [w.herd_weight for w in monitor.path_weights_history],
                    'ai': [w.ai_weight for w in monitor.path_weights_history]
                }
                for key, line in weight_lines.items():
                    line.set_data(time_data, weight_data[key])
                monitor_axes[1, 0].relim()
                monitor_axes[1, 0].autoscale_view()
            
            # 更新速度图表
            if len(monitor.speed_history) > 0:
                monitor_lines['speed'].set_data(time_data, monitor.speed_history)
                monitor_axes[1, 1].relim()
                monitor_axes[1, 1].autoscale_view()
            
            if monitor_fig is not None:
                monitor_fig.canvas.draw_idle()

        return_items = [scat_adult, scat_elder, scat_child, stats_text]
        if scat4 is not None:
            return_items.append(scat4)
        return tuple(return_items)

    ani = FuncAnimation(fig, update, frames=300, interval=100, blit=False)
    
    # 如果启用了监视器，也更新监视器窗口
    if monitor is not None and monitor_fig is not None:
        def update_monitor(frame):
            # 监视器窗口通过主update函数中的图表更新逻辑更新
            # 这里返回空列表，因为实际更新在主update函数中完成
            return []
        monitor_ani = FuncAnimation(monitor_fig, update_monitor, frames=300, interval=100, blit=False)
        
    plt.legend()
    plt.show()  # 显示主仿真窗口

    return ani
