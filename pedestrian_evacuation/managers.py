"""行人疏散模拟系统 - 管理器类（处理业务逻辑）"""

import numpy as np
import heapq
from typing import List, Optional, Tuple
from . import constants
from .entities import AgentData, ObstacleData, DangerData, AreaData, FireData
from .models import Map


class EnvironmentManager:
    """环境管理器：处理障碍物和危险源的交互"""
    
    @staticmethod
    def calculate_obstacle_acceleration(obstacle: ObstacleData, pos: np.ndarray) -> np.ndarray:
        """计算障碍物对位置的影响加速度"""
        acc = np.zeros(2)
        vec_dir = pos - obstacle.pos
        dist = np.linalg.norm(vec_dir) if np.linalg.norm(vec_dir) > 1e-3 else 1e-3
        acc += obstacle.A * np.exp(-dist / obstacle.B) * vec_dir / dist
        return acc
    
    @staticmethod
    def calculate_danger_level(danger: DangerData, pos: np.ndarray) -> float:
        """计算危险源在位置的危险程度"""
        dist = np.linalg.norm(pos - danger.pos)
        return np.exp(-dist / danger.L) + danger.S
    
    @staticmethod
    def calculate_environmental_acceleration(
        obstacles: List[ObstacleData],
        dangers: List[DangerData],
        pos: np.ndarray
    ) -> np.ndarray:
        """计算环境（障碍物+危险源）对位置的总加速度"""
        acc = np.zeros(2)
        for obstacle in obstacles:
            acc += EnvironmentManager.calculate_obstacle_acceleration(obstacle, pos)
        for danger in dangers:
            # 直接使用danger对象的A和B参数（对于火灾，这些参数已经被设置为火灾特有的值）
            # 创建一个临时ObstacleData对象，但使用danger的参数
            temp_obstacle = ObstacleData(danger.pos, "danger")
            temp_obstacle.A = danger.A  # 使用danger的A参数（火灾使用FIRE_ENVIRONMENTAL_FORCE_A）
            temp_obstacle.B = danger.B  # 使用danger的B参数（火灾使用FIRE_ENVIRONMENTAL_FORCE_B）
            acc += EnvironmentManager.calculate_obstacle_acceleration(temp_obstacle, pos)
        return acc
    
    @staticmethod
    def get_max_danger_level(dangers: List[DangerData], pos: np.ndarray) -> float:
        """获取位置的最大危险程度"""
        if not dangers:
            return 0.0
        return max(EnvironmentManager.calculate_danger_level(d, pos) for d in dangers)


class BoundaryManager:
    """边界管理器：处理区域边界力"""
    
    @staticmethod
    def calculate_boundary_force(
        area: AreaData,
        pos: np.ndarray,
        vel: np.ndarray,
        tau: float
    ) -> np.ndarray:
        """计算边界对位置和速度的力"""
        left, lower = area.left_bottom
        right, upper = area.right_top
        x, y = pos
        acc = np.zeros(2)
        
        # 处理每个方向的边界
        boundary_configs = [
            ('left', x - left, 0, left, lambda v: v < 0),
            ('right', right - x, 0, right, lambda v: v > 0),
            ('upper', upper - y, 1, upper, lambda v: v > 0),
            ('lower', y - lower, 1, lower, lambda v: v < 0)
        ]
        
        for bound_name, dist, axis, bound_pos, vel_check in boundary_configs:
            if bound_name in area.jammed:
                if dist < 0:
                    # 在边界外，强制推回
                    acc[axis] = (bound_pos - pos[axis]) / tau * 300
                elif dist < 0.5:
                    # 接近边界，推离（力度随距离减小而增强）
                    force_magnitude = 150 * (1.0 - dist / 0.5)
                    acc[axis] = float(force_magnitude / tau)
                    # 如果速度朝向边界，额外增加阻力
                    if vel_check(vel[axis]):
                        acc[axis] = float(acc[axis] - vel[axis] * 50 / tau)
        
        return acc


class PhysicsEngine:
    """物理引擎：处理所有物理计算"""
    
    @staticmethod
    def calculate_interpersonal_acceleration(
        agent: AgentData,
        other: AgentData
    ) -> np.ndarray:
        """计算两个行人之间的人际加速度"""
        acc = np.zeros(2)
        vec_dir = agent.pos - other.pos
        dist_sq = np.dot(vec_dir, vec_dir)
        dist = np.sqrt(dist_sq) if dist_sq > 1e-6 else 1e-3
        acc += (agent.A * vec_dir / dist * np.exp(-dist / agent.B) + 
                agent.C * agent.vel / (dist + 1))
        return acc
    
    @staticmethod
    def calculate_goal_acceleration(
        agent: AgentData,
        target: np.ndarray
    ) -> np.ndarray:
        """计算目标加速度"""
        vec_to_target = target - agent.pos
        dist_to_target = np.linalg.norm(vec_to_target)
        
        if dist_to_target < 1e-3:
            # 已经非常接近目标，减速
            acc_goal = -constants.ATTRACTION * agent.vel / agent.reaction_time
        else:
            vel_desired = vec_to_target / dist_to_target * agent.desired_speed
            acc_goal = constants.ATTRACTION * (vel_desired - agent.vel) / agent.reaction_time
        
        return acc_goal
    
    @staticmethod
    def calculate_psychological_acceleration(agent: AgentData) -> np.ndarray:
        """计算心理状态加速度"""
        activated_panic = np.tanh(agent.panic)
        panic_direction = np.random.uniform(-1, 1, 2)
        panic_direction = (panic_direction / np.linalg.norm(panic_direction) 
                          if np.linalg.norm(panic_direction) > 1e-3 
                          else np.array([1.0, 0.0]))
        acc_psy = activated_panic * agent.panic_probability_factor * panic_direction
        return acc_psy
    
    @staticmethod
    def calculate_total_acceleration(
        agent: AgentData,
        target: np.ndarray,
        neighbors: List[AgentData],
        obstacles: List[ObstacleData],
        dangers: List[DangerData],
        boundary_area: Optional[AreaData],
        return_components: bool = False
    ):
        """计算总加速度
        
        Args:
            return_components: 如果为True，返回分解的加速度组件和总加速度
        
        Returns:
            如果return_components=False，返回总加速度（np.ndarray）
            如果return_components=True，返回元组 (acc_components_dict, acc_total)
        """
        # 目标加速度
        acc_goal = PhysicsEngine.calculate_goal_acceleration(agent, target)
        
        # 人际加速度
        acc_int = np.zeros(2)
        for neighbor in neighbors:
            if neighbor is not agent:
                acc_int += PhysicsEngine.calculate_interpersonal_acceleration(agent, neighbor)
        
        # 环境加速度（分解为障碍物和危险源）
        acc_obstacle = EnvironmentManager.calculate_environmental_acceleration(
            obstacles, [], agent.pos
        )
        
        # 分离普通危险源和火灾
        acc_danger = np.zeros(2)
        acc_fire = np.zeros(2)
        for danger in dangers:
            # 检查是否是火灾（通过dan_type或S值判断）
            is_fire = False
            if hasattr(danger, 'dan_type') and danger.dan_type == "fire":
                is_fire = True
            elif hasattr(danger, 'S') and danger.S >= 2.0:  # 火灾的stimuli值较高
                is_fire = True
            
            # 使用danger的A和B参数计算加速度
            temp_obstacle = ObstacleData(danger.pos, "danger")
            temp_obstacle.A = danger.A
            temp_obstacle.B = danger.B
            danger_acc = EnvironmentManager.calculate_obstacle_acceleration(temp_obstacle, agent.pos)
            
            if is_fire:
                acc_fire += danger_acc
            else:
                acc_danger += danger_acc
        
        # 边界加速度
        acc_bound = np.zeros(2)
        if boundary_area is not None:
            acc_bound = BoundaryManager.calculate_boundary_force(
                boundary_area, agent.pos, agent.vel, agent.reaction_time
            )
        
        # 心理加速度
        acc_psy = PhysicsEngine.calculate_psychological_acceleration(agent)
        
        # 总加速度
        acc_total = acc_goal + acc_int + acc_obstacle + acc_danger + acc_fire + acc_bound + acc_psy
        
        if return_components:
            from .monitor import AccelerationRecord
            acc_record = AccelerationRecord(
                acc_goal=acc_goal,
                acc_int=acc_int,
                acc_obstacle=acc_obstacle,
                acc_danger=acc_danger,
                acc_fire=acc_fire,
                acc_bound=acc_bound,
                acc_psy=acc_psy,
                acc_total=acc_total
            )
            return acc_record, acc_total
        else:
            return acc_total


class MPCPlanner:
    """基于MPC（模型预测控制）的AI路径推荐器
    
    使用MPC算法预测未来路径上的人流密度和危险程度，为行人推荐最优路径。
    """
    
    def __init__(self, map_manager: Map):
        self.map_manager = map_manager
        self.last_update_time = {}  # {agent_id: last_update_time} 记录每个agent的上次更新时间
        self.recommendations = {}  # {agent_id: recommended_area_id} 存储AI推荐
    
    def predict_path_density(
        self,
        path: List[int],
        all_agents: List[AgentData],
        current_time: float,
        prediction_horizon: int
    ) -> float:
        """预测路径上未来的人流密度
        
        Args:
            path: 路径（区域ID列表）
            all_agents: 所有行人列表
            current_time: 当前时间
            prediction_horizon: 预测时域（步数）
        
        Returns:
            预测的平均人流密度
        """
        if not path:
            return 0.0
        
        total_density = 0.0
        for area_id in path[:prediction_horizon]:
            if area_id not in self.map_manager.areas:
                continue
            
            area = self.map_manager.areas[area_id]
            area_center = np.array(area.center)
            
            # 计算当前在该区域的行人数量
            density = 0
            for agent in all_agents:
                if agent.evacuated:
                    continue
                # 预测agent未来可能的位置（简单预测：当前位置+速度*时间）
                predicted_pos = agent.pos + agent.vel * current_time * 0.1  # 简单预测
                if area.inside(predicted_pos) or area.inside(agent.pos):
                    density += 1
            
            # 归一化密度（除以区域面积）
            area_width = area.right_top[0] - area.left_bottom[0]
            area_height = area.right_top[1] - area.left_bottom[1]
            area_size = area_width * area_height if area_width > 0 and area_height > 0 else 1.0
            normalized_density = density / max(area_size, 1.0)
            total_density += normalized_density
        
        return total_density / len(path[:prediction_horizon]) if path[:prediction_horizon] else 0.0
    
    def predict_path_danger(
        self,
        path: List[int],
        dangers: List[DangerData],
        fire_manager: Optional['FireManager'],
        prediction_horizon: int
    ) -> float:
        """预测路径上未来的危险程度
        
        Args:
            path: 路径（区域ID列表）
            dangers: 危险源列表
            fire_manager: 火灾管理器（可选）
            prediction_horizon: 预测时域（步数）
        
        Returns:
            预测的平均危险程度
        """
        if not path:
            return 0.0
        
        total_danger = 0.0
        for area_id in path[:prediction_horizon]:
            if area_id not in self.map_manager.areas:
                continue
            
            area = self.map_manager.areas[area_id]
            area_center = np.array(area.center)
            
            # 计算该区域的危险程度
            area_danger = 0.0
            
            # 检查普通危险源
            for danger in dangers:
                danger_dist = np.linalg.norm(area_center - danger.pos)
                danger_level = EnvironmentManager.calculate_danger_level(danger, area_center)
                area_danger += danger_level / (1.0 + danger_dist)
            
            # 检查火灾（如果启用）
            if fire_manager is not None:
                if fire_manager.is_area_on_fire(area_id):
                    area_danger += 10.0  # 火灾区域危险值很高
                # 检查相邻区域是否有火灾（火灾可能扩散）
                adjacent_areas = self.map_manager.get_adjacent_areas(area_id)
                for adj_area in adjacent_areas:
                    if fire_manager.is_area_on_fire(adj_area.id):
                        area_danger += 5.0  # 相邻区域有火灾，危险值增加
            
            total_danger += area_danger
        
        return total_danger / len(path[:prediction_horizon]) if path[:prediction_horizon] else 0.0
    
    def compute_path_cost(
        self,
        path: List[int],
        exit_pos: np.ndarray,
        all_agents: List[AgentData],
        dangers: List[DangerData],
        fire_manager: Optional['FireManager'],
        current_time: float,
        prediction_horizon: int
    ) -> float:
        """计算路径的代价（MPC代价函数）
        
        代价 = 距离代价 + 密度代价 + 危险代价
        
        Args:
            path: 路径（区域ID列表）
            exit_pos: 出口位置
            all_agents: 所有行人列表
            dangers: 危险源列表
            fire_manager: 火灾管理器（可选）
            current_time: 当前时间
            prediction_horizon: 预测时域
        
        Returns:
            路径代价（越小越好）
        """
        if not path:
            return float('inf')
        
        # 1. 距离代价
        if path:
            last_area = self.map_manager.areas.get(path[-1])
            if last_area:
                end_pos = np.array(last_area.center)
                distance = np.linalg.norm(end_pos - exit_pos)
            else:
                distance = float('inf')
        else:
            distance = float('inf')
        distance_cost = distance * constants.MPC_DISTANCE_WEIGHT
        
        # 2. 密度代价
        density = self.predict_path_density(path, all_agents, current_time, prediction_horizon)
        density_cost = density * constants.MPC_DENSITY_WEIGHT * 10.0  # 放大密度影响
        
        # 3. 危险代价
        danger = self.predict_path_danger(path, dangers, fire_manager, prediction_horizon)
        danger_cost = danger * constants.MPC_DANGER_WEIGHT * 5.0  # 放大危险影响
        
        total_cost = distance_cost + density_cost + danger_cost
        return total_cost
    
    def recommend_path(
        self,
        agent: AgentData,
        choices: List[int],
        exit_pos: np.ndarray,
        all_agents: List[AgentData],
        dangers: List[DangerData],
        fire_manager: Optional['FireManager'],
        current_time: float
    ) -> Optional[int]:
        """使用MPC算法推荐最优路径
        
        Args:
            agent: 当前行人
            choices: 可选择的区域ID列表
            exit_pos: 出口位置
            all_agents: 所有行人列表
            dangers: 危险源列表
            fire_manager: 火灾管理器（可选）
            current_time: 当前时间
        
        Returns:
            推荐的区域ID，如果没有推荐则返回None
        """
        if not choices or self.map_manager is None:
            return None
        
        # 检查是否需要更新推荐（根据更新间隔）
        agent_id = id(agent)
        if agent_id in self.last_update_time:
            time_since_update = current_time - self.last_update_time[agent_id]
            if time_since_update < constants.MPC_UPDATE_INTERVAL:
                # 使用缓存的推荐
                return self.recommendations.get(agent_id)
        
        # 计算每个选择的代价
        choice_costs = {}
        # 创建PathPlanner实例用于路径查找
        path_planner = PathPlanner(self.map_manager)
        
        for area_id in choices:
            if area_id not in self.map_manager.areas:
                continue
            
            # 找到从该区域到出口的路径
            path = path_planner.find_path(area_id, agent.exit_node_id)
            if not path:
                # 如果无法到达出口，代价设为无穷大
                choice_costs[area_id] = float('inf')
                continue
            
            # 计算路径代价
            cost = self.compute_path_cost(
                path, exit_pos, all_agents, dangers, fire_manager,
                current_time, constants.MPC_HORIZON
            )
            choice_costs[area_id] = cost
        
        # 选择代价最小的路径
        if not choice_costs:
            return None
        
        # 找到最小代价
        min_cost = min(choice_costs.values())
        if min_cost == float('inf'):
            return None
        
        # 选择代价最小的区域（如果有多个，随机选择一个）
        best_choices = [aid for aid, cost in choice_costs.items() if cost == min_cost]
        recommended = np.random.choice(best_choices) if best_choices else None
        
        # 缓存推荐
        self.last_update_time[agent_id] = current_time
        if recommended is not None:
            self.recommendations[agent_id] = recommended
        
        return recommended


class PathPlanner:
    """路径规划器：处理路径规划和选择"""
    
    def __init__(self, map_manager: Map):
        self.map_manager = map_manager
    
    def find_path(self, start_area_id: int, end_area_id: int) -> List[int]:
        """使用A*算法查找路径"""
        if start_area_id not in self.map_manager.areas or end_area_id not in self.map_manager.areas:
            return []
        
        if start_area_id == end_area_id:
            return [start_area_id]
        
        # A*算法
        open_set_heap = [(0.0, start_area_id)]
        open_set_dict = {start_area_id: True}
        came_from = {}
        g_score = {start_area_id: 0.0}
        f_score = {}
        
        start_area = self.map_manager.areas[start_area_id]
        end_area = self.map_manager.areas[end_area_id]
        start_center = np.array(start_area.center)
        end_center = np.array(end_area.center)
        h_start = float(np.linalg.norm(start_center - end_center))
        f_score[start_area_id] = h_start
        
        while open_set_heap:
            current_f, current_id = heapq.heappop(open_set_heap)
            if current_id not in open_set_dict:
                continue
            del open_set_dict[current_id]
            
            if current_id == end_area_id:
                # 重构路径
                path = [current_id]
                while current_id in came_from:
                    current_id = came_from[current_id]
                    path.insert(0, current_id)
                return path
            
            current_area = self.map_manager.areas[current_id]
            current_center = np.array(current_area.center)
            
            for neighbor_id in current_area.adjacent:
                if neighbor_id not in self.map_manager.areas:
                    continue
                
                neighbor_area = self.map_manager.areas[neighbor_id]
                
                if not self.map_manager.is_area_passable(neighbor_id):
                    continue
                
                neighbor_center = np.array(neighbor_area.center)
                tentative_g_score = float(g_score[current_id] + np.linalg.norm(neighbor_center - current_center))
                
                if neighbor_id not in g_score or tentative_g_score < g_score[neighbor_id]:
                    came_from[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g_score
                    h_score = float(np.linalg.norm(neighbor_center - end_center))
                    f_score[neighbor_id] = tentative_g_score + h_score
                    if neighbor_id not in open_set_dict:
                        heapq.heappush(open_set_heap, (f_score[neighbor_id], neighbor_id))
                        open_set_dict[neighbor_id] = True
        
        return []
    
    def choose_exit(
        self,
        agent: AgentData,
        exits: List[tuple],  # List of (exit_node_id, exit_pos)
        neighbors: List[AgentData],
        dangers: List[DangerData],
        obstacles: List[ObstacleData],
        all_agents: Optional[List[AgentData]] = None,
        mpc_planner: Optional['MPCPlanner'] = None,
        fire_manager: Optional['FireManager'] = None,
        current_time: float = 0.0
    ) -> Optional[tuple]:
        """选择最优出口：考虑距离、拥挤度、安全性和路径长度
        
        Args:
            agent: 当前行人
            exits: 可用出口列表，每个元素为 (exit_node_id, exit_pos)
            neighbors: 邻居行人列表
            dangers: 危险源列表
            obstacles: 障碍物列表
            all_agents: 所有行人列表（用于计算拥挤度）
            mpc_planner: MPC规划器（可选）
            fire_manager: 火灾管理器（可选）
            current_time: 当前时间
            
        Returns:
            最优出口 (exit_node_id, exit_pos)，如果没有可用出口则返回None
        """
        if not exits or self.map_manager is None:
            return None
        
        # 获取当前位置所在区域
        current_area = self.map_manager.get_area_containing_point(agent.pos)
        if current_area is None:
            # 如果无法确定当前位置，随机选择一个出口
            return exits[np.random.randint(0, len(exits))] if exits else None
        
        current_area_id = current_area.id
        
        # 获取该行人的心理阈值
        threshold = constants.STIMULUS_THRESHOLD.get(
            agent.agent_type, 
            constants.STIMULUS_THRESHOLD_DEFAULT
        )
        # 恐慌值超过阈值时，随机选择出口
        if agent.panic > threshold:
            return exits[np.random.randint(0, len(exits))] if exits else None
        
        # 简单模式：选择距离最近的出口
        if not agent.use_smart_choice:
            min_dist = float('inf')
            chosen_exit = None
            for exit_node_id, exit_pos in exits:
                if exit_node_id not in self.map_manager.areas:
                    continue
                dist = float(np.linalg.norm(agent.pos - exit_pos))
                if dist < min_dist:
                    min_dist = dist
                    chosen_exit = (exit_node_id, exit_pos)
            return chosen_exit if chosen_exit else (exits[0] if exits else None)
        
        # 智能模式：计算每个出口的得分
        exit_scores = {}
        all_agents = all_agents or []
        
        for exit_node_id, exit_pos in exits:
            if exit_node_id not in self.map_manager.areas:
                continue
            
            # 1. 距离得分（考虑路径长度，而不仅仅是直线距离）
            path = self.find_path(current_area_id, exit_node_id)
            if not path:
                # 如果无法到达该出口，跳过
                continue
            
            # 计算路径总长度
            path_length = 0.0
            for i in range(len(path) - 1):
                area1 = self.map_manager.areas[path[i]]
                area2 = self.map_manager.areas[path[i + 1]]
                path_length += np.linalg.norm(np.array(area2.center) - np.array(area1.center))
            
            # 加上到出口位置的距离
            if exit_node_id in self.map_manager.areas:
                exit_area = self.map_manager.areas[exit_node_id]
                path_length += np.linalg.norm(exit_pos - np.array(exit_area.center))
            
            distance_score = 1.0 / (1.0 + path_length)
            
            # 2. 拥挤度得分（出口附近的行人数量）
            crowd_score = 1.0
            if all_agents:
                crowd_radius = 3.0  # 考虑出口周围3米内的行人
                nearby_agents = 0
                for other_agent in all_agents:
                    if other_agent is agent or other_agent.evacuated:
                        continue
                    dist_to_exit = np.linalg.norm(other_agent.pos - exit_pos)
                    if dist_to_exit < crowd_radius:
                        nearby_agents += 1
                
                # 拥挤度越高，得分越低
                crowd_score = 1.0 / (1.0 + nearby_agents * 0.3)
            
            # 3. 危险程度得分
            danger_score = 1.0
            if dangers:
                total_danger = 0.0
                for danger in dangers:
                    danger_dist = np.linalg.norm(exit_pos - danger.pos)
                    danger_level = EnvironmentManager.calculate_danger_level(danger, exit_pos)
                    total_danger += danger_level / (1.0 + danger_dist)
                danger_score = 1.0 / (1.0 + total_danger * constants.DANGER_WEIGHT)
            
            # 4. 障碍物影响得分
            obstacle_score = 1.0
            if obstacles:
                total_obstacle = 0.0
                threshold = 2.0
                for obstacle in obstacles:
                    obs_dist = np.linalg.norm(exit_pos - obstacle.pos)
                    if obs_dist < threshold:
                        total_obstacle += 1.0 / (1.0 + obs_dist)
                obstacle_score = 1.0 / (1.0 + total_obstacle * 0.5)
            
            # 5. 从众行为得分（有多少邻居选择了这个出口）
            herd_score = 1.0
            if neighbors and constants.HERD_BEHAVIOR_FACTOR > 0:
                neighbor_exits = 0
                total_neighbors = 0
                for neighbor in neighbors:
                    if neighbor.exit_node_id != -1:
                        total_neighbors += 1
                        if neighbor.exit_node_id == exit_node_id:
                            neighbor_exits += 1
                
                if total_neighbors > 0:
                    exit_ratio = neighbor_exits / total_neighbors
                    herd_score = 1.0 + constants.HERD_BEHAVIOR_FACTOR * exit_ratio
            
            # 6. AI推荐得分（如果启用MPC，可以推荐出口）
            ai_score = 1.0
            if (constants.MPC_ENABLED and mpc_planner is not None and 
                all_agents is not None):
                # 使用MPC推荐路径到该出口
                # 这里简化处理，如果MPC推荐了到该出口的路径，给予更高得分
                # 实际实现中，可以调用MPC来评估每个出口
                pass  # 暂时不实现，因为MPC主要针对路径选择
            
            # 综合得分（权重受panic影响）
            activated_panic = np.tanh(agent.panic)
            panic_weight_factor = activated_panic * agent.panic_probability_factor
            
            base_weights = {
                'distance': 0.3,
                'crowd': 0.25,  # 拥挤度权重
                'danger': 0.25,
                'obstacle': 0.1,
                'herd': 0.1
            }
            
            distance_weight = base_weights['distance'] * (1.0 - panic_weight_factor * 0.3)
            crowd_weight = base_weights['crowd'] * (1.0 - panic_weight_factor * 0.2)
            danger_weight = base_weights['danger'] * (1.0 + panic_weight_factor * 0.5)
            obstacle_weight = base_weights['obstacle'] * (1.0 + panic_weight_factor * 0.4)
            herd_weight = base_weights['herd'] * (1.0 - panic_weight_factor * 0.2)
            
            # 归一化权重
            total_weight = (distance_weight + crowd_weight + danger_weight + 
                          obstacle_weight + herd_weight)
            if total_weight > 1e-6:
                distance_weight /= total_weight
                crowd_weight /= total_weight
                danger_weight /= total_weight
                obstacle_weight /= total_weight
                herd_weight /= total_weight
            
            total_score = (distance_score * distance_weight + 
                          crowd_score * crowd_weight +
                          danger_score * danger_weight + 
                          obstacle_score * obstacle_weight + 
                          herd_score * herd_weight)
            exit_scores[(exit_node_id, tuple(exit_pos))] = total_score
        
        if len(exit_scores) == 0:
            # 如果没有可用出口，返回第一个
            return exits[0] if exits else None
        
        # 使用概率选择（softmax）
        exit_tuples = list(exit_scores.keys())
        score_values = np.array([exit_scores[et] for et in exit_tuples])
        exp_scores = np.exp(score_values * 5.0)
        probabilities = exp_scores / np.sum(exp_scores)
        chosen_idx = np.random.choice(len(exit_tuples), p=probabilities)
        chosen_exit_tuple = exit_tuples[chosen_idx]
        
        # 转换回原始格式
        exit_node_id = chosen_exit_tuple[0]
        exit_pos = np.array(chosen_exit_tuple[1])
        return (exit_node_id, exit_pos)
    
    def choose_path(
        self,
        agent: AgentData,
        exit_pos: np.ndarray,
        choices: List[int],
        neighbors: List[AgentData],
        dangers: List[DangerData],
        obstacles: List[ObstacleData],
        mpc_planner: Optional['MPCPlanner'] = None,
        all_agents: Optional[List[AgentData]] = None,
        fire_manager: Optional['FireManager'] = None,
        current_time: float = 0.0
    ) -> Optional[int]:
        """路径选择：简单模式选择最近节点，智能模式考虑危险和从众行为，可选AI推荐"""
        if not choices or self.map_manager is None:
            return None
        
        # 验证选项的有效性（检查区域是否存在且可通行）
        valid_choices = [aid for aid in choices 
                        if aid in self.map_manager.areas 
                        and self.map_manager.is_area_passable(aid)]
        if not valid_choices:
            return None
        
        # 获取该行人的心理阈值
        threshold = constants.STIMULUS_THRESHOLD.get(
            agent.agent_type, 
            constants.STIMULUS_THRESHOLD_DEFAULT
        )
        # 恐慌值超过阈值时，直接从当前选项中随机选择
        if agent.panic > threshold:
            # 设置默认权重（用于监视器）
            from .monitor import PathSelectionWeights
            path_weights = PathSelectionWeights(
                distance_weight=0.0,  # 恐慌时随机选择，不考虑权重
                danger_weight=0.0,
                obstacle_weight=0.0,
                herd_weight=0.0,
                ai_weight=0.0
            )
            if not hasattr(agent, '_path_weights'):
                agent._path_weights = None
            agent._path_weights = path_weights
            return np.random.choice(valid_choices)
        
        # 简单模式：选择距离出口最近的节点
        if not agent.use_smart_choice:
            min_dist = float('inf')
            chosen_id = None
            for aid in valid_choices:
                area = self.map_manager.areas[aid]
                dist = float(np.linalg.norm(np.array(area.center) - exit_pos))
                if dist < min_dist:
                    min_dist = dist
                    chosen_id = aid
            
            # 设置默认权重（用于监视器）
            from .monitor import PathSelectionWeights
            path_weights = PathSelectionWeights(
                distance_weight=1.0,  # 简单模式只考虑距离
                danger_weight=0.0,
                obstacle_weight=0.0,
                herd_weight=0.0,
                ai_weight=0.0
            )
            if not hasattr(agent, '_path_weights'):
                agent._path_weights = None
            agent._path_weights = path_weights
            
            return chosen_id
        
        # 获取AI推荐（如果启用）
        ai_recommendation = None
        if (constants.MPC_ENABLED and mpc_planner is not None and 
            all_agents is not None):
            ai_recommendation = mpc_planner.recommend_path(
                agent, valid_choices, exit_pos, all_agents, 
                dangers, fire_manager, current_time
            )
        
        # 智能模式：计算每个选择的得分
        area_scores = {}
        
        for area_id in choices:
            if area_id not in self.map_manager.areas:
                continue
            
            # 确保区域可通行（双重检查，确保安全）
            if not self.map_manager.is_area_passable(area_id):
                continue
            
            area = self.map_manager.areas[area_id]
            area_center = np.array(area.center)
            
            # 1. 距离得分
            dist_to_exit = np.linalg.norm(area_center - exit_pos)
            distance_score = 1.0 / (1.0 + dist_to_exit)
            
            # 2. 危险程度得分
            danger_score = 1.0
            if dangers:
                total_danger = 0.0
                for danger in dangers:
                    danger_dist = np.linalg.norm(area_center - danger.pos)
                    danger_level = EnvironmentManager.calculate_danger_level(danger, area_center)
                    total_danger += danger_level / (1.0 + danger_dist)
                danger_score = 1.0 / (1.0 + total_danger * constants.DANGER_WEIGHT)
            
            # 3. 障碍物影响得分
            obstacle_score = 1.0
            if obstacles:
                total_obstacle = 0.0
                threshold = 2.0
                for obstacle in obstacles:
                    obs_dist = np.linalg.norm(area_center - obstacle.pos)
                    if obs_dist < threshold:
                        total_obstacle += 1.0 / (1.0 + obs_dist)
                obstacle_score = 1.0 / (1.0 + total_obstacle * 0.5)
            
            # 4. 从众行为得分
            herd_score = 1.0
            if neighbors and constants.HERD_BEHAVIOR_FACTOR > 0:
                neighbor_choices = 0
                total_neighbors = 0
                for neighbor in neighbors:
                    # 使用committed_target_area_id来判断从众行为（更准确，反映实际移动目标）
                    neighbor_target_id = neighbor.committed_target_area_id if neighbor.committed_target_area_id != -1 else neighbor.next_target_area_id
                    if neighbor_target_id != -1:
                        total_neighbors += 1
                        if neighbor_target_id == area_id:
                            neighbor_choices += 1
                
                if total_neighbors > 0:
                    choice_ratio = neighbor_choices / total_neighbors
                    herd_score = 1.0 + constants.HERD_BEHAVIOR_FACTOR * choice_ratio
            
            # 5. AI推荐得分（如果启用）
            ai_score = 1.0
            if ai_recommendation is not None:
                if area_id == ai_recommendation:
                    ai_score = 2.0  # AI推荐的路径得分更高
                else:
                    ai_score = 0.5  # 非AI推荐的路径得分较低
            
            # 综合得分（权重受panic和概率因子影响）
            activated_panic = np.tanh(agent.panic)
            panic_weight_factor = activated_panic * agent.panic_probability_factor
            
            base_weights = {
                'distance': 0.25,
                'danger': 0.35,
                'obstacle': 0.15,
                'herd': 0.1,
                'ai': 0.15  # AI推荐权重
            }
            
            # 如果未启用AI，将AI权重分配给其他项
            if not constants.MPC_ENABLED or ai_recommendation is None:
                base_weights['distance'] += base_weights['ai'] * 0.4
                base_weights['danger'] += base_weights['ai'] * 0.4
                base_weights['obstacle'] += base_weights['ai'] * 0.2
                base_weights['ai'] = 0.0
            
            distance_weight = base_weights['distance'] * (1.0 - panic_weight_factor * 0.3)
            danger_weight = base_weights['danger'] * (1.0 + panic_weight_factor * 0.5)
            obstacle_weight = base_weights['obstacle'] * (1.0 + panic_weight_factor * 0.4)
            herd_weight = base_weights['herd'] * (1.0 - panic_weight_factor * 0.2)
            ai_weight = base_weights['ai'] * constants.MPC_AI_RECOMMENDATION_WEIGHT
            
            # 归一化权重
            total_weight = (distance_weight + danger_weight + obstacle_weight + 
                          herd_weight + ai_weight)
            if total_weight > 1e-6:
                distance_weight /= total_weight
                danger_weight /= total_weight
                obstacle_weight /= total_weight
                herd_weight /= total_weight
                ai_weight /= total_weight
            
            total_score = (distance_score * distance_weight + 
                          danger_score * danger_weight + 
                          obstacle_score * obstacle_weight + 
                          herd_score * herd_weight +
                          ai_score * ai_weight)
            area_scores[area_id] = total_score
        
        # 存储路径选择权重（用于监视器）- 使用最后一次计算的权重
        if area_scores:
            from .monitor import PathSelectionWeights
            path_weights = PathSelectionWeights(
                distance_weight=distance_weight,
                danger_weight=danger_weight,
                obstacle_weight=obstacle_weight,
                herd_weight=herd_weight,
                ai_weight=ai_weight
            )
            if not hasattr(agent, '_path_weights'):
                agent._path_weights = None
            agent._path_weights = path_weights
        
        if len(area_scores) == 0:
            return None
        
        # 使用概率选择
        area_ids = list(area_scores.keys())
        score_values = np.array([area_scores[aid] for aid in area_ids])
        exp_scores = np.exp(score_values * 5.0)
        probabilities = exp_scores / np.sum(exp_scores)
        chosen_id = np.random.choice(area_ids, p=probabilities)
        return chosen_id
    
    def update_path(
        self,
        agent: AgentData,
        neighbors: List[AgentData],
        dangers: List[DangerData],
        obstacles: List[ObstacleData],
        mpc_planner: Optional['MPCPlanner'] = None,
        all_agents: Optional[List[AgentData]] = None,
        fire_manager: Optional['FireManager'] = None,
        current_time: float = 0.0
    ) -> None:
        """更新路径规划"""
        if agent.exit_node_id == -1:
            return
        
        # 查找当前所在区域
        current_area = self.map_manager.get_area_containing_point(agent.pos)
        
        # 如果不在任何区域内或不可通行，找到最近的可通行区域
        if current_area is None or not self.map_manager.is_area_passable(current_area.id):
            nearest_area = self.map_manager.find_nearest_passable_area(agent.pos)
            if nearest_area is None:
                return
            if current_area is None:
                current_area = nearest_area
            else:
                agent.target_pos = np.array(nearest_area.center)
                agent.path = []
                return
        
        current_area_id = current_area.id
        
        # 如果路径为空或当前区域不在路径中，重新规划路径
        if len(agent.path) == 0 or current_area_id not in agent.path:
            agent.path = self.find_path(current_area_id, agent.exit_node_id)
            agent.path_index = 0
            # 路径重新规划时，如果committed_target_area_id不在新路径中，重置它
            if agent.committed_target_area_id != -1:
                if agent.committed_target_area_id not in agent.path and agent.committed_target_area_id != agent.exit_node_id:
                    agent.committed_target_area_id = -1
        
        # 如果路径规划失败，直接朝向出口
        if len(agent.path) == 0:
            if agent.exit_node_id in self.map_manager.areas:
                exit_area = self.map_manager.areas[agent.exit_node_id]
                agent.target_pos = np.array(exit_area.center)
                agent.next_target_area_id = agent.exit_node_id
            return
        
        # 找到当前区域在路径中的位置
        try:
            agent.path_index = agent.path.index(current_area_id)
        except ValueError:
            agent.path = self.find_path(current_area_id, agent.exit_node_id)
            if len(agent.path) > 0:
                agent.path_index = 0
            else:
                return
        
        # 如果已经到达路径中的最后一个区域（出口）
        if agent.path_index >= len(agent.path) - 1:
            if agent.exit_node_id in self.map_manager.areas:
                exit_area = self.map_manager.areas[agent.exit_node_id]
                agent.next_target_area_id = agent.exit_node_id
                # 只有当next_target_area_id与committed_target_area_id相同时，才更新承诺的ID
                if agent.exit_node_id == agent.committed_target_area_id:
                    if agent.exit_pos is not None:
                        agent.target_pos = agent.exit_pos
                    else:
                        agent.target_pos = np.array(exit_area.center)
                # 如果committed_target_area_id未设置（初始状态），则立即设置
                elif agent.committed_target_area_id == -1:
                    agent.committed_target_area_id = agent.exit_node_id
                    if agent.exit_pos is not None:
                        agent.target_pos = agent.exit_pos
                    else:
                        agent.target_pos = np.array(exit_area.center)
        else:
            # 目标指向路径中的下一个区域
            if agent.use_smart_choice and current_area.is_node():
                adjacent_areas = self.map_manager.get_adjacent_areas(current_area_id)
                if len(adjacent_areas) > 0:
                    # 只选择可通行的区域，且该区域在路径中或是出口
                    choices = [area.id for area in adjacent_areas 
                              if (self.map_manager.is_area_passable(area.id) and
                                  (area.id in agent.path or area.id == agent.exit_node_id))]
                    if len(choices) > 0:
                        if agent.exit_pos is not None:
                            exit_pos = agent.exit_pos
                        elif agent.exit_node_id in self.map_manager.areas:
                            exit_area = self.map_manager.areas[agent.exit_node_id]
                            exit_pos = np.array(exit_area.center)
                        else:
                            exit_pos = np.array([0, 0])
                        
                        chosen_id = self.choose_path(
                            agent, exit_pos, choices, neighbors, dangers, obstacles,
                            mpc_planner, all_agents, fire_manager, current_time
                        )
                        if chosen_id is not None:
                            # 只有当选择的ID与已承诺的ID相同时，才更新承诺的ID
                            if chosen_id == agent.committed_target_area_id:
                                # 更新next_target_area_id用于路径规划
                                agent.next_target_area_id = chosen_id
                                try:
                                    chosen_idx = agent.path.index(chosen_id)
                                    if chosen_idx > agent.path_index:
                                        agent.path_index = chosen_idx - 1
                                except ValueError:
                                    pass
                            else:
                                # 选择的ID与承诺的ID不同，只更新next_target_area_id（用于规划），不更新承诺的ID
                                agent.next_target_area_id = chosen_id
            
            # 动态更新目标点为路径中下一个节点的中心点
            next_area_id = (agent.path[agent.path_index + 1] 
                          if agent.path_index + 1 < len(agent.path) 
                          else agent.exit_node_id)
            if next_area_id in self.map_manager.areas:
                next_area = self.map_manager.areas[next_area_id]
                agent.next_target_area_id = next_area_id
                # 只有当next_target_area_id与committed_target_area_id相同时，才更新承诺的ID
                if next_area_id == agent.committed_target_area_id:
                    # 目标点始终设置为下一个节点的中心点
                    agent.target_pos = np.array(next_area.center)
                # 如果committed_target_area_id未设置（初始状态），则立即设置
                elif agent.committed_target_area_id == -1:
                    agent.committed_target_area_id = next_area_id
                    agent.target_pos = np.array(next_area.center)
            else:
                # 如果下一个区域不存在，尝试使用出口
                if agent.exit_node_id in self.map_manager.areas:
                    exit_area = self.map_manager.areas[agent.exit_node_id]
                    agent.next_target_area_id = agent.exit_node_id
                    # 只有当next_target_area_id与committed_target_area_id相同时，才更新承诺的ID
                    if agent.exit_node_id == agent.committed_target_area_id:
                        if agent.exit_pos is not None:
                            agent.target_pos = agent.exit_pos
                        else:
                            agent.target_pos = np.array(exit_area.center)
                    # 如果committed_target_area_id未设置（初始状态），则立即设置
                    elif agent.committed_target_area_id == -1:
                        agent.committed_target_area_id = agent.exit_node_id
                        if agent.exit_pos is not None:
                            agent.target_pos = agent.exit_pos
                        else:
                            agent.target_pos = np.array(exit_area.center)


class AgentManager:
    """Agent管理器：管理Agent的状态更新"""
    
    def __init__(self, map_manager: Map):
        self.map_manager = map_manager
        self.path_planner = PathPlanner(map_manager)
    
    def update_panic(
        self,
        agent: AgentData,
        neighbors: List[AgentData],
        dangers: List[DangerData]
    ) -> None:
        """更新panic值"""
        perceive_radius_sq = agent.perceive_radius ** 2
        neighbor_count = sum(1 for n in neighbors 
                           if np.dot(agent.pos - n.pos, agent.pos - n.pos) < perceive_radius_sq)
        danger = EnvironmentManager.get_max_danger_level(dangers, agent.pos)
        agent.panic = 0.1 * neighbor_count / np.pi + danger
        
        # 生成新的正态分布概率因子
        agent.panic_probability_factor = np.random.normal(
            constants.PANIC_PROBABILITY_MEAN, 
            constants.PANIC_PROBABILITY_STD
        )
        agent.panic_probability_factor = max(0.1, agent.panic_probability_factor)
    
    def update_state(
        self,
        agent: AgentData,
        neighbors: List[AgentData],
        obstacles: List[ObstacleData],
        dangers: List[DangerData],
        boundary_area: Optional[AreaData],
        current_time: float = 0.0,
        mpc_planner: Optional['MPCPlanner'] = None,
        all_agents: Optional[List[AgentData]] = None,
        fire_manager: Optional['FireManager'] = None
    ) -> None:
        """更新Agent状态"""
        if agent.evacuated:
            return
        
        # 更新路径规划
        self.path_planner.update_path(
            agent, neighbors, dangers, obstacles,
            mpc_planner, all_agents, fire_manager, current_time
        )
        
        # 获取当前位置所在区域（用于智能目标点选择）
        current_area = None
        if self.map_manager is not None:
            current_area = self.map_manager.get_area_containing_point(agent.pos)
        
        # 动态更新目标点
        if len(agent.path) > 0 and agent.path_index < len(agent.path):
            # 检查是否使用智能逻辑选择目标点
            if (constants.SMART_TARGET_SELECTION and agent.use_smart_choice and 
                current_area is not None and current_area.is_node()):
                # 使用智能逻辑选择目标点（与路径选择逻辑相同）
                adjacent_areas = self.map_manager.get_adjacent_areas(current_area.id)
                if len(adjacent_areas) > 0:
                    # 获取所有可选的相邻区域（包括路径中的和出口），只选择可通行的区域
                    choices = [area.id for area in adjacent_areas 
                              if (self.map_manager.is_area_passable(area.id) and
                                  (area.id in agent.path or area.id == agent.exit_node_id))]
                    if len(choices) > 0:
                        if agent.exit_pos is not None:
                            exit_pos = agent.exit_pos
                        elif agent.exit_node_id in self.map_manager.areas:
                            exit_area = self.map_manager.areas[agent.exit_node_id]
                            exit_pos = np.array(exit_area.center)
                        else:
                            exit_pos = np.array([0, 0])
                        
                        # 使用智能选择逻辑选择目标点
                        chosen_id = self.path_planner.choose_path(
                            agent, exit_pos, choices, neighbors, dangers, obstacles,
                            mpc_planner, all_agents, fire_manager, current_time
                        )
                        if chosen_id is not None and chosen_id in self.map_manager.areas:
                            chosen_area = self.map_manager.areas[chosen_id]
                            agent.next_target_area_id = chosen_id
                            # 只有当选择的ID与已承诺的ID相同时，才更新承诺的ID和目标点
                            if chosen_id == agent.committed_target_area_id:
                                agent.target_pos = np.array(chosen_area.center)
                            # 如果committed_target_area_id未设置（初始状态），则立即设置
                            elif agent.committed_target_area_id == -1:
                                agent.committed_target_area_id = chosen_id
                                agent.target_pos = np.array(chosen_area.center)
                        else:
                            # 如果智能选择失败，回退到路径中的下一个节点
                            next_idx = agent.path_index + 1
                            if next_idx < len(agent.path):
                                next_area_id = agent.path[next_idx]
                                if next_area_id in self.map_manager.areas:
                                    next_area = self.map_manager.areas[next_area_id]
                                    agent.next_target_area_id = next_area_id
                                    # 只有当next_target_area_id与committed_target_area_id相同时，才更新承诺的ID
                                    if next_area_id == agent.committed_target_area_id:
                                        agent.target_pos = np.array(next_area.center)
                                    # 如果committed_target_area_id未设置（初始状态），则立即设置
                                    elif agent.committed_target_area_id == -1:
                                        agent.committed_target_area_id = next_area_id
                                        agent.target_pos = np.array(next_area.center)
                            elif agent.exit_node_id in self.map_manager.areas:
                                agent.next_target_area_id = agent.exit_node_id
                                # 只有当next_target_area_id与committed_target_area_id相同时，才更新承诺的ID
                                if agent.exit_node_id == agent.committed_target_area_id:
                                    if agent.exit_pos is not None:
                                        agent.target_pos = agent.exit_pos
                                    else:
                                        exit_area = self.map_manager.areas[agent.exit_node_id]
                                        agent.target_pos = np.array(exit_area.center)
                                # 如果committed_target_area_id未设置（初始状态），则立即设置
                                elif agent.committed_target_area_id == -1:
                                    agent.committed_target_area_id = agent.exit_node_id
                                    if agent.exit_pos is not None:
                                        agent.target_pos = agent.exit_pos
                                    else:
                                        exit_area = self.map_manager.areas[agent.exit_node_id]
                                        agent.target_pos = np.array(exit_area.center)
                    else:
                        # 没有可选区域，使用路径中的下一个节点
                        next_idx = agent.path_index + 1
                        if next_idx < len(agent.path):
                            next_area_id = agent.path[next_idx]
                            if next_area_id in self.map_manager.areas:
                                next_area = self.map_manager.areas[next_area_id]
                                agent.next_target_area_id = next_area_id
                                # 只有当next_target_area_id与committed_target_area_id相同时，才更新承诺的ID
                                if next_area_id == agent.committed_target_area_id:
                                    agent.target_pos = np.array(next_area.center)
                                # 如果committed_target_area_id未设置（初始状态），则立即设置
                                elif agent.committed_target_area_id == -1:
                                    agent.committed_target_area_id = next_area_id
                                    agent.target_pos = np.array(next_area.center)
            else:
                # 默认行为：目标点设置为路径中下一个节点的中心点
                next_idx = agent.path_index + 1
                if next_idx < len(agent.path):
                    next_area_id = agent.path[next_idx]
                    if next_area_id in self.map_manager.areas:
                        next_area = self.map_manager.areas[next_area_id]
                        agent.next_target_area_id = next_area_id
                        # 只有当next_target_area_id与committed_target_area_id相同时，才更新承诺的ID
                        if next_area_id == agent.committed_target_area_id:
                            agent.target_pos = np.array(next_area.center)
                        # 如果committed_target_area_id未设置（初始状态），则立即设置
                        elif agent.committed_target_area_id == -1:
                            agent.committed_target_area_id = next_area_id
                            agent.target_pos = np.array(next_area.center)
                elif agent.exit_node_id in self.map_manager.areas:
                    # 如果已经到达路径末尾，目标指向出口
                    agent.next_target_area_id = agent.exit_node_id
                    # 只有当next_target_area_id与committed_target_area_id相同时，才更新承诺的ID
                    if agent.exit_node_id == agent.committed_target_area_id:
                        if agent.exit_pos is not None:
                            agent.target_pos = agent.exit_pos
                        else:
                            exit_area = self.map_manager.areas[agent.exit_node_id]
                            agent.target_pos = np.array(exit_area.center)
                    # 如果committed_target_area_id未设置（初始状态），则立即设置
                    elif agent.committed_target_area_id == -1:
                        agent.committed_target_area_id = agent.exit_node_id
                        if agent.exit_pos is not None:
                            agent.target_pos = agent.exit_pos
                        else:
                            exit_area = self.map_manager.areas[agent.exit_node_id]
                            agent.target_pos = np.array(exit_area.center)
        
        # 确定目标点（用于计算加速度）
        # 使用committed_target_area_id来确定目标点，而不是next_target_area_id
        if agent.committed_target_area_id != -1 and self.map_manager is not None:
            if agent.committed_target_area_id in self.map_manager.areas:
                committed_area = self.map_manager.areas[agent.committed_target_area_id]
                if agent.committed_target_area_id == agent.exit_node_id and agent.exit_pos is not None:
                    target = agent.exit_pos
                else:
                    target = np.array(committed_area.center)
            else:
                target = agent.target_pos if agent.target_pos is not None else agent.pos
        else:
            target = agent.target_pos if agent.target_pos is not None else agent.pos
        
        # 更新panic
        self.update_panic(agent, neighbors, dangers)
        
        # 计算加速度（如果需要监视，返回分解的组件）
        acc_record, acc = PhysicsEngine.calculate_total_acceleration(
            agent, target, neighbors, obstacles, dangers, boundary_area,
            return_components=True
        )
        
        # 存储加速度记录（用于监视器）
        if not hasattr(agent, '_acc_record'):
            agent._acc_record = None
        agent._acc_record = acc_record
        
        # 更新速度
        agent.vel += acc * constants.SIMULATION_DT
        if np.linalg.norm(agent.vel) > constants.MAX_SPEED:
            agent.vel = agent.vel / np.linalg.norm(agent.vel) * constants.MAX_SPEED
        
        # 更新位置
        agent.pos += agent.vel * constants.SIMULATION_DT
        
        # 检查是否在不可通行区域内，如果是则强制推回
        if self.map_manager is not None:
            current_area = self.map_manager.get_area_containing_point(agent.pos)
            if current_area is not None and not self.map_manager.is_area_passable(current_area.id):
                nearest_area = self.map_manager.find_nearest_passable_area(agent.pos)
                if nearest_area is not None:
                    push_dir = np.array(nearest_area.center) - agent.pos
                    push_dist = float(np.linalg.norm(push_dir))
                    if push_dist > 1e-3:
                        push_dir = push_dir / push_dist
                        agent.pos += push_dir * min(0.2, push_dist * 0.5)
        
        # 检查是否到达撤离点
        if agent.exit_pos is not None and isinstance(agent.exit_pos, np.ndarray):
            dist_to_exit = np.linalg.norm(agent.pos - agent.exit_pos)
            if dist_to_exit < 0.5:
                agent.evacuated = True
                return
        
        # 检查是否到达路径中的中间目标（使用committed_target_area_id）
        if agent.committed_target_area_id != -1 and self.map_manager is not None:
            if agent.committed_target_area_id in self.map_manager.areas:
                committed_area = self.map_manager.areas[agent.committed_target_area_id]
                # 检查是否到达已承诺的目标区域
                if self._point_in_area(agent.pos, committed_area):
                    # 到达目标区域，更新committed_target_area_id为下一个目标
                    # 使用next_target_area_id作为新的承诺目标（如果已设置）
                    if agent.next_target_area_id != -1 and agent.next_target_area_id != agent.committed_target_area_id:
                        # 更新承诺的目标ID
                        agent.committed_target_area_id = agent.next_target_area_id
                        if agent.next_target_area_id in self.map_manager.areas:
                            next_area = self.map_manager.areas[agent.next_target_area_id]
                            if agent.next_target_area_id == agent.exit_node_id and agent.exit_pos is not None:
                                agent.target_pos = agent.exit_pos
                            else:
                                agent.target_pos = np.array(next_area.center)
                        # 更新路径索引
                        if len(agent.path) > 0 and agent.path_index < len(agent.path) - 1:
                            try:
                                new_index = agent.path.index(agent.next_target_area_id)
                                if new_index > agent.path_index:
                                    agent.path_index = new_index
                            except ValueError:
                                pass
                    elif agent.committed_target_area_id == agent.exit_node_id:
                        # 已到达出口区域，但还需要到达exit_pos
                        if agent.exit_pos is not None:
                            agent.target_pos = agent.exit_pos
                else:
                    # 未到达目标区域，检查距离（作为备用判断）
                    if agent.target_pos is not None and isinstance(agent.target_pos, np.ndarray):
                        dist_to_target = np.linalg.norm(agent.pos - agent.target_pos)
                        if dist_to_target < 0.3:
                            # 距离很近，认为已到达
                            if agent.next_target_area_id != -1 and agent.next_target_area_id != agent.committed_target_area_id:
                                agent.committed_target_area_id = agent.next_target_area_id
                                if agent.next_target_area_id in self.map_manager.areas:
                                    next_area = self.map_manager.areas[agent.next_target_area_id]
                                    if agent.next_target_area_id == agent.exit_node_id and agent.exit_pos is not None:
                                        agent.target_pos = agent.exit_pos
                                    else:
                                        agent.target_pos = np.array(next_area.center)
                                if len(agent.path) > 0 and agent.path_index < len(agent.path) - 1:
                                    try:
                                        new_index = agent.path.index(agent.next_target_area_id)
                                        if new_index > agent.path_index:
                                            agent.path_index = new_index
                                    except ValueError:
                                        pass
    
    @staticmethod
    def _point_in_area(pos: np.ndarray, area: AreaData) -> bool:
        """判断点是否在区域内"""
        x, y = pos
        left, bottom = area.left_bottom
        right, top = area.right_top
        return left < x < right and bottom < y < top


class FireManager:
    """火灾管理器：使用元胞自动机管理火灾的扩散和熄灭"""
    
    def __init__(self, map_manager: Map):
        self.map_manager = map_manager
        self.fires: List[FireData] = []  # 当前活跃的火灾列表
        self.fire_areas: dict = {}  # {area_id: FireData} 快速查找某个区域是否有火灾
        self.extinguished_areas: set = set()  # 已熄灭的火灾区域ID集合（用于可视化）
    
    def add_fire(self, area_id: int, current_time: float) -> Optional[FireData]:
        """在指定区域添加火灾"""
        if area_id not in self.map_manager.areas:
            return None
        
        # 检查该区域是否已经有火灾
        if area_id in self.fire_areas:
            return self.fire_areas[area_id]
        
        # 检查区域是否可通行（火灾只能在可通行区域）
        if not self.map_manager.is_area_passable(area_id):
            return None
        
        import random
        from . import constants
        
        # 创建火灾
        fire = FireData(
            area_id=area_id,
            start_time=current_time,
            spread_time=random.uniform(constants.FIRE_SPREAD_TIME_MIN, 
                                       constants.FIRE_SPREAD_TIME_MAX),
            duration=random.uniform(constants.FIRE_DURATION_MIN, 
                                   constants.FIRE_DURATION_MAX)
        )
        fire.next_spread_time = current_time + fire.spread_time
        
        self.fires.append(fire)
        self.fire_areas[area_id] = fire
        return fire
    
    def remove_fire(self, area_id: int) -> bool:
        """移除指定区域的火灾（添加到已熄灭集合）"""
        if area_id not in self.fire_areas:
            return False
        
        fire = self.fire_areas[area_id]
        self.fires.remove(fire)
        del self.fire_areas[area_id]
        # 添加到已熄灭集合，用于可视化
        self.extinguished_areas.add(area_id)
        return True
    
    def update(self, current_time: float) -> List[int]:
        """更新火灾状态（扩散和熄灭），返回新着火的区域ID列表"""
        new_fires = []
        fires_to_remove = []
        
        for fire in self.fires:
            # 检查是否应该熄灭
            if current_time >= fire.start_time + fire.duration:
                fires_to_remove.append(fire.area_id)
                continue
            
            # 检查是否应该扩散
            if current_time >= fire.next_spread_time:
                # 尝试扩散到相邻区域
                adjacent_areas = self.map_manager.get_adjacent_areas(fire.area_id)
                valid_targets = [
                    area.id for area in adjacent_areas
                    if (self.map_manager.is_area_passable(area.id) and 
                        area.id not in self.fire_areas and
                        area.id not in fire.spread_targets)
                ]
                
                if valid_targets:
                    import random
                    from . import constants
                    
                    # 根据概率选择是否扩散
                    if random.random() < constants.FIRE_SPREAD_PROBABILITY:
                        # 随机选择一个相邻区域扩散
                        target_area_id = random.choice(valid_targets)
                        fire.spread_targets.append(target_area_id)
                        
                        # 创建新火灾
                        new_fire = FireData(
                            area_id=target_area_id,
                            start_time=current_time,
                            spread_time=random.uniform(constants.FIRE_SPREAD_TIME_MIN,
                                                      constants.FIRE_SPREAD_TIME_MAX),
                            duration=random.uniform(constants.FIRE_DURATION_MIN,
                                                  constants.FIRE_DURATION_MAX)
                        )
                        new_fire.next_spread_time = current_time + new_fire.spread_time
                        
                        self.fires.append(new_fire)
                        self.fire_areas[target_area_id] = new_fire
                        new_fires.append(target_area_id)
                
                # 更新下次扩散时间
                import random
                from . import constants
                fire.spread_time = random.uniform(constants.FIRE_SPREAD_TIME_MIN,
                                                 constants.FIRE_SPREAD_TIME_MAX)
                fire.next_spread_time = current_time + fire.spread_time
        
        # 移除已熄灭的火灾
        for area_id in fires_to_remove:
            self.remove_fire(area_id)
        
        return new_fires
    
    def get_fire_positions(self) -> List[np.ndarray]:
        """获取所有火灾的位置（区域中心）"""
        positions = []
        for fire in self.fires:
            if fire.area_id in self.map_manager.areas:
                area = self.map_manager.areas[fire.area_id]
                positions.append(np.array(area.center))
        return positions
    
    def get_fire_dangers(self) -> List[DangerData]:
        """获取所有火灾作为危险源（用于影响行人）"""
        dangers = []
        for fire in self.fires:
            if fire.area_id in self.map_manager.areas:
                area = self.map_manager.areas[fire.area_id]
                # 创建火灾危险源
                fire_danger = DangerData(
                    pos=np.array(area.center),
                    dan_type="fire"
                )
                # 设置火灾特有的参数
                from . import constants
                fire_danger.A = constants.FIRE_ENVIRONMENTAL_FORCE_A
                fire_danger.B = constants.FIRE_ENVIRONMENTAL_FORCE_B
                fire_danger.L = constants.FIRE_DANGER_LAMBDA
                fire_danger.S = constants.FIRE_DANGER_STIMULI
                dangers.append(fire_danger)
        return dangers
    
    def is_area_on_fire(self, area_id: int) -> bool:
        """检查指定区域是否着火"""
        return area_id in self.fire_areas
    
    def get_fire_count(self) -> int:
        """获取当前火灾数量"""
        return len(self.fires)
    
    def is_area_extinguished(self, area_id: int) -> bool:
        """检查指定区域是否曾经着火但已熄灭"""
        return area_id in self.extinguished_areas
    
    def clear_all_fires(self):
        """清除所有火灾"""
        self.fires.clear()
        self.fire_areas.clear()
        self.extinguished_areas.clear()
