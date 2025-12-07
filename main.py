"""行人疏散模拟系统 - 主入口文件"""

from pedestrian_evacuation import run_simulation

if __name__ == "__main__":
    # ========== 基本仿真 ==========
    ani = run_simulation(use_smart_choice=True,
    num_agents=20,
    num_obstacles=5,
    num_danger=2,
    enable_fire=True,
    fire_initial_count=2,
    enable_ai=True,
    ai_recommendation_weight=0.5,
    mpc_horizon=5,
    mpc_update_interval=1.0,
    monitor_agent_index=0)
    
    # ========== 示例：监视器系统 ==========
    # 监视第一个行人（索引0），显示实时数据图表
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     num_obstacles=10,
    #     num_danger=5,
    #     monitor_agent_index=0  # 监视第一个行人
    # )
    
    # 监视行人 + 火灾系统
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_fire=True,
    #     fire_initial_count=2,
    #     monitor_agent_index=0  # 监视第一个行人，观察火灾对其影响
    # )
    
    # 监视行人 + AI路径推荐
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_ai=True,
    #     ai_recommendation_weight=0.5,
    #     monitor_agent_index=0  # 监视第一个行人，观察AI推荐的影响
    # )
    
    # ========== 示例：火灾系统 ==========
    # 启用火灾系统，使用默认参数
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_fire=True
    # )
    
    # 启用火灾系统，自定义参数
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     num_obstacles=10,
    #     num_danger=5,
    #     enable_fire=True,
    #     fire_initial_count=2,  # 初始火灾数量
    #     fire_spawn_probability=0.001,  # 每帧生成火灾的概率（0.1%）
    #     fire_spread_time_min=2.0,  # 火灾扩散最小时间（秒）
    #     fire_spread_time_max=5.0,  # 火灾扩散最大时间（秒）
    #     fire_duration_min=10.0,  # 火灾持续最小时间（秒）
    #     fire_duration_max=20.0,  # 火灾持续最大时间（秒）
    #     fire_spread_probability=0.6  # 火灾扩散到相邻区域的概率（60%）
    # )
    
    # ========== 示例：AI路径推荐系统 ==========
    # 启用AI路径推荐，使用默认参数
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_ai=True
    # )
    
    # 启用AI路径推荐，自定义参数
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     num_obstacles=10,
    #     num_danger=5,
    #     enable_ai=True,
    #     ai_recommendation_weight=0.4,  # AI推荐权重（40%）
    #     mpc_horizon=5,  # 预测未来5步
    #     mpc_update_interval=1.0  # 每1秒更新一次推荐
    # )
    
    # AI + 火灾系统
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_fire=True,
    #     fire_initial_count=2,
    #     enable_ai=True,
    #     ai_recommendation_weight=0.5,  # 高权重，更信任AI
    #     mpc_horizon=7  # 更长的预测时域
    # )
    
    # 注意：动画对象会保持窗口打开，直到用户关闭窗口
