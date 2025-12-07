"""监视器使用示例

展示如何使用监视器实时监控特定行人的状态。
"""

from pedestrian_evacuation import run_simulation

if __name__ == "__main__":
    # 示例1：监视第一个行人（索引0）
    print("示例1：监视第一个行人")
    ani = run_simulation(
        use_smart_choice=True,
        num_agents=20,
        num_obstacles=10,
        num_danger=5,
        monitor_agent_index=0  # 监视第一个行人
    )
    
    # 示例2：监视行人 + 火灾系统
    print("示例2：监视行人 + 火灾系统")
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_fire=True,
    #     fire_initial_count=2,
    #     monitor_agent_index=0  # 监视第一个行人，观察火灾对其影响
    # )
    
    # 示例3：监视行人 + AI路径推荐
    print("示例3：监视行人 + AI路径推荐")
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     enable_ai=True,
    #     ai_recommendation_weight=0.5,
    #     monitor_agent_index=0  # 监视第一个行人，观察AI推荐的影响
    # )
    
    # 示例4：监视特定行人（索引5）
    print("示例4：监视特定行人（索引5）")
    # ani = run_simulation(
    #     use_smart_choice=True,
    #     num_agents=20,
    #     monitor_agent_index=5  # 监视第6个行人（索引从0开始）
    # )
