"""AI路径推荐系统使用示例

展示如何使用基于MPC的AI路径推荐系统。
"""

from pedestrian_evacuation import run_simulation

if __name__ == "__main__":
    # 示例1：启用AI路径推荐，使用默认参数
    print("示例1：启用AI路径推荐（默认参数）")
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_ai=True
    # )
    
    # 示例2：启用AI路径推荐，自定义参数
    print("示例2：启用AI路径推荐（自定义参数）")
    ani = run_simulation(
        use_smart_choice=True,
        num_agents=20,
        num_obstacles=10,
        num_danger=5,
        enable_ai=True,
        # AI推荐参数
        ai_recommendation_weight=0.4,  # AI推荐权重（40%）
        mpc_horizon=5,  # 预测未来5步
        mpc_update_interval=1.0  # 每1秒更新一次推荐
    )
    
    # 示例3：AI + 火灾系统
    print("示例3：AI路径推荐 + 火灾系统")
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_fire=True,
    #     fire_initial_count=2,
    #     enable_ai=True,
    #     ai_recommendation_weight=0.5,  # 高权重，更信任AI
    #     mpc_horizon=7  # 更长的预测时域
    # )
    
    # 示例4：高AI权重（更信任AI推荐）
    print("示例4：高AI权重（更信任AI推荐）")
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_ai=True,
    #     ai_recommendation_weight=0.7,  # 70%权重，高度信任AI
    #     mpc_horizon=6,
    #     mpc_update_interval=0.5  # 更频繁的更新
    # )
    
    # 示例5：低AI权重（AI仅作为参考）
    print("示例5：低AI权重（AI仅作为参考）")
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_ai=True,
    #     ai_recommendation_weight=0.2,  # 20%权重，AI仅作为参考
    #     mpc_horizon=4,
    #     mpc_update_interval=2.0  # 较少的更新
    # )
