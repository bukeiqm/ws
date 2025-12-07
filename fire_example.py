"""火灾系统使用示例

展示如何使用火灾系统，包括调试参数的设置。
"""

from pedestrian_evacuation import run_simulation

if __name__ == "__main__":
    # 示例1：启用火灾系统，使用默认参数
    print("示例1：启用火灾系统（默认参数）")
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_fire=True
    # )
    
    # 示例2：启用火灾系统，自定义调试参数
    print("示例2：启用火灾系统（自定义参数）")
    ani = run_simulation(
        use_smart_choice=True,
        num_agents=20,
        num_obstacles=10,
        num_danger=5,
        enable_fire=True,
        # 火灾调试参数
        fire_initial_count=2,  # 初始火灾数量
        fire_spawn_probability=0.001,  # 每帧生成火灾的概率（0.1%）
        fire_spread_time_min=2.0,  # 火灾扩散最小时间（秒）
        fire_spread_time_max=5.0,  # 火灾扩散最大时间（秒）
        fire_duration_min=10.0,  # 火灾持续最小时间（秒）
        fire_duration_max=20.0,  # 火灾持续最大时间（秒）
        fire_spread_probability=0.6  # 火灾扩散到相邻区域的概率（60%）
    )
    
    # 示例3：快速扩散的火灾（用于测试）
    print("示例3：快速扩散的火灾（测试用）")
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=15,
    #     enable_fire=True,
    #     fire_initial_count=1,
    #     fire_spread_time_min=1.0,  # 快速扩散
    #     fire_spread_time_max=2.0,
    #     fire_duration_min=15.0,  # 持续时间较长
    #     fire_duration_max=25.0,
    #     fire_spread_probability=0.8  # 高扩散概率
    # )
    
    # 示例4：缓慢扩散的火灾
    print("示例4：缓慢扩散的火灾")
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=15,
    #     enable_fire=True,
    #     fire_initial_count=1,
    #     fire_spread_time_min=5.0,  # 缓慢扩散
    #     fire_spread_time_max=8.0,
    #     fire_duration_min=8.0,  # 持续时间较短
    #     fire_duration_max=12.0,
    #     fire_spread_probability=0.3  # 低扩散概率
    # )
