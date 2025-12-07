"""行人疏散模拟系统 - 主入口文件"""

from pedestrian_evacuation import run_simulation

if __name__ == "__main__":
    # 运行仿真
    ani = run_simulation(use_smart_choice=None, 
                    num_agents=20, 
                    num_obstacles=10, 
                    num_danger=5,
                    enable_fire=True, 
                    fire_initial_count=2, 
                  fire_spawn_probability=0.001,
                  fire_spread_time_min=2.0, 
                  fire_spread_time_max=5.0,
                  fire_duration_min=10.0, 
                  fire_duration_max=20.0,
                  fire_spread_probability=0.6,
                  enable_ai=True, 
                  ai_recommendation_weight=0.4,
                  mpc_horizon=5, 
                  mpc_update_interval=1.0,
                  monitor_agent_index=0)
    # 注意：动画对象会保持窗口打开，直到用户关闭窗口
