"""行人疏散模拟系统核心模型类"""

import numpy as np
from . import constants
from .entities import AreaData


# 为了向后兼容，保留Obs和Dan作为别名
# 但建议使用entities中的ObstacleData和DangerData
from .entities import ObstacleData as Obs, DangerData as Dan


# 为了向后兼容，保留Agent作为别名
# 但建议使用entities中的AgentData和managers中的AgentManager
from .entities import AgentData as Agent


class Area:
    """行人（智能体）类"""
    def __init__(self, pos, agent_type="adult", map_manager=None):
        if pos is not None:
            self.pos = np.array(pos, dtype=float)
        else:
            self.pos = np.random.rand(2)
        self.vel = np.zeros(2)
        self.panic = 0.0
        self.desired_speed = constants.DESIRED_SPEED[agent_type]
        self.reaction_time = constants.REACTION_TIME[agent_type]
        self.safe_distance = constants.SAFE_DISTANCE[agent_type]
        self.perceive_radius = constants.PERCEIVE_RADIUS[agent_type]
        self.A = constants.INTERPERSONAL_FORCE_PARAMS[agent_type]["A"]
        self.B = constants.INTERPERSONAL_FORCE_PARAMS[agent_type]["B"]
        self.C = constants.HERD_FACT if agent_type == "adult" else constants.HERD_FACT_NORMAL
        self.evacuated = False
        self.current_area_id = -1
        self.exit_node_id = -1
        self.exit_pos = None  # 可以是None或np.ndarray
        self.map_manager = map_manager
        self.target_pos = None
        self.path = []
        self.path_index = 0
        self.next_target_area_id = -1
        self.use_smart_choice = True
        self.panic_probability_factor = 1.0  # 当前更新的概率因子（正态分布）

    def choice_smart(self, exit_pos, choices, neighbors=None, dangers=None, obstacles=None):
        """路径选择：简单模式选择最近节点，智能模式考虑危险和从众行为"""
        if not choices or self.map_manager is None:
            return None
        
        # 简单模式或恐慌时随机选择
        if not self.use_smart_choice or self.panic > constants.STIMULUS_THRESHOLD:
            valid_choices = [aid for aid in choices if aid in self.map_manager.areas]
            if not valid_choices:
                return None
            if self.panic > constants.STIMULUS_THRESHOLD:
                return np.random.choice(valid_choices)
            # 简单模式：选择距离出口最近的节点
            min_dist = float('inf')
            chosen_id = None
            for aid in valid_choices:
                area = self.map_manager.areas[aid]
                dist = float(np.linalg.norm(np.array(area.center) - exit_pos))
                if dist < min_dist:
                    min_dist = dist
                    chosen_id = aid
            return chosen_id
        
        neighbors = neighbors or []
        dangers = dangers or []
        obstacles = obstacles or []
        
        # 计算每个选择的得分
        area_scores = {}
        
        for area_id in choices:
            if area_id not in self.map_manager.areas:
                continue
            
            area = self.map_manager.areas[area_id]
            area_center = np.array(area.center)
            
            # 1. 距离得分（距离越近得分越高）
            dist_to_exit = np.linalg.norm(area_center - exit_pos)
            distance_score = 1.0 / (1.0 + dist_to_exit)
            
            # 2. 危险程度得分（危险越多得分越低）
            danger_score = 1.0
            if dangers:
                total_danger = 0.0
                for danger in dangers:
                    danger_dist = np.linalg.norm(area_center - danger.pos)
                    danger_level = danger.danger(area_center)
                    total_danger += danger_level / (1.0 + danger_dist)
                danger_score = 1.0 / (1.0 + total_danger * constants.DANGER_WEIGHT)
            
            # 3. 障碍物影响（障碍物越多得分越低）
            obstacle_score = 1.0
            if obstacles:
                total_obstacle = 0.0
                threshold = 2.0
                for obstacle in obstacles:
                    obs_dist = np.linalg.norm(area_center - obstacle.pos)
                    if obs_dist < threshold:  # 只考虑附近的障碍物
                        total_obstacle += 1.0 / (1.0 + obs_dist)
                obstacle_score = 1.0 / (1.0 + total_obstacle * 0.5)
            
            # 4. 从众行为得分（附近的人选择的路径得分更高）
            herd_score = 1.0
            if neighbors and constants.HERD_BEHAVIOR_FACTOR > 0:
                neighbor_choices = 0
                total_neighbors = 0
                for neighbor in neighbors:
                    if hasattr(neighbor, 'next_target_area_id') and neighbor.next_target_area_id != -1:
                        total_neighbors += 1
                        if neighbor.next_target_area_id == area_id:
                            neighbor_choices += 1
                
                if total_neighbors > 0:
                    choice_ratio = neighbor_choices / total_neighbors
                    herd_score = 1.0 + constants.HERD_BEHAVIOR_FACTOR * choice_ratio
            
            # 综合得分（权重受panic和概率因子影响）
            # 使用激活函数作用于panic，然后乘以概率因子来调整权重
            activated_panic = np.tanh(self.panic)
            panic_weight_factor = activated_panic * self.panic_probability_factor
            
            # 基础权重
            base_weights = {
                'distance': 0.3,
                'danger': 0.4,
                'obstacle': 0.2,
                'herd': 0.1
            }
            
            # 应用panic影响：panic越高，对危险和障碍物的权重增加，对距离和从众的权重减少
            # 使用乘算方式：权重 = 基础权重 * (1 + panic_weight_factor * 调整系数)
            distance_weight = base_weights['distance'] * (1.0 - panic_weight_factor * 0.3)
            danger_weight = base_weights['danger'] * (1.0 + panic_weight_factor * 0.5)
            obstacle_weight = base_weights['obstacle'] * (1.0 + panic_weight_factor * 0.4)
            herd_weight = base_weights['herd'] * (1.0 - panic_weight_factor * 0.2)
            
            # 归一化权重（确保总和为1）
            total_weight = distance_weight + danger_weight + obstacle_weight + herd_weight
            if total_weight > 1e-6:
                distance_weight /= total_weight
                danger_weight /= total_weight
                obstacle_weight /= total_weight
                herd_weight /= total_weight
            
            total_score = (distance_score * distance_weight + 
                          danger_score * danger_weight + 
                          obstacle_score * obstacle_weight + 
                          herd_score * herd_weight)
            area_scores[area_id] = total_score
        
        if len(area_scores) == 0:
            return None
        
        # 使用概率选择（得分越高被选中的概率越大）
        area_ids = list(area_scores.keys())
        score_values = np.array([area_scores[aid] for aid in area_ids])
        
        # 转换为概率（使用softmax）
        exp_scores = np.exp(score_values * 5.0)  # 放大差异
        probabilities = exp_scores / np.sum(exp_scores)
        
        # 根据概率随机选择
        chosen_id = np.random.choice(area_ids, p=probabilities)
        return chosen_id

    def acc_int(self, other, A, B):
        """Get interpersonal acc (优化：使用平方距离避免重复开方)"""
        C = self.C
        acc = np.zeros(2)
        # distance and direction as well as velocity
        vec_dir = self.pos - other.pos
        dist_sq = np.dot(vec_dir, vec_dir)
        dist = np.sqrt(dist_sq) if dist_sq > 1e-6 else 1e-3
        acc += A * vec_dir / dist * np.exp(-dist / B) + C * self.vel / (dist + 1)
        return acc

    def calc_acc(self, target, others, obstacles, dangers, bound):
        # Accleration from goal
        vec_to_target = target - self.pos
        dist_to_target = np.linalg.norm(vec_to_target)
        
        # 如果已经非常接近目标，减小吸引力
        if dist_to_target < 1e-3:
            acc_goal = -constants.ATTRACTION * self.vel / self.reaction_time  # 减速
        else:
            vel_desired = vec_to_target / dist_to_target * self.desired_speed
            acc_goal = constants.ATTRACTION * (vel_desired - self.vel) / self.reaction_time

        # Accleration from other person
        acc_int = np.zeros(2)
        for each in others:
            acc_int += self.acc_int(each, self.A, self.B) if each is not self else np.zeros(2)

        # Accleration from environment (obstacles and dangers)
        acc_env = np.zeros(2)
        for each in (obstacles + dangers):
            acc_env += each.acc_env(self.pos)

        # Accleration from boundaries
        acc_bound = np.zeros(2)
        if bound is not None:
            acc_bound = bound.force(self.pos, self.vel, self.reaction_time)

        # Accleration from mental state
        # 使用激活函数（tanh）作用于panic，然后乘以正态分布的概率因子
        activated_panic = np.tanh(self.panic)  # 激活函数作用于panic
        # 心理加速度 = 激活后的panic * 概率因子（方向随机）
        panic_direction = np.random.uniform(-1, 1, 2)
        panic_direction = panic_direction / np.linalg.norm(panic_direction) if np.linalg.norm(panic_direction) > 1e-3 else np.array([1.0, 0.0])
        acc_psy = activated_panic * self.panic_probability_factor * panic_direction

        total_acc = acc_goal + acc_int + acc_env + acc_bound + acc_psy
        return total_acc
    
    def update_path(self, neighbors=None, dangers=None, obstacles=None):
        """更新路径规划"""
        if self.map_manager is None or self.exit_node_id == -1:
            return
        
        # 查找当前所在区域
        current_area = self.map_manager.get_area_containing_point(self.pos)
        
        # 如果不在任何区域内或不可通行，找到最近的可通行区域
        if current_area is None or not self.map_manager.is_area_passable(current_area.id):
            nearest_area = self.map_manager.find_nearest_passable_area(self.pos)
            if nearest_area is None:
                return
            if current_area is None:
                current_area = nearest_area
            else:
                # 在不可通行区域内，朝向最近的可通行区域移动
                self.target_pos = np.array(nearest_area.center)
                self.path = []
                return
        
        current_area_id = current_area.id
        
        # 如果路径为空或当前区域不在路径中，重新规划路径
        if len(self.path) == 0 or current_area_id not in self.path:
            self.path = self.map_manager.find_path(current_area_id, self.exit_node_id)
            self.path_index = 0
        
        # 如果路径规划失败，直接朝向出口
        if len(self.path) == 0:
            if self.exit_node_id in self.map_manager.areas:
                exit_area = self.map_manager.areas[self.exit_node_id]
                self.target_pos = np.array(exit_area.center)
                self.next_target_area_id = self.exit_node_id
            return
        
        # 如果路径规划成功
        # 找到当前区域在路径中的位置
        try:
            self.path_index = self.path.index(current_area_id)
        except ValueError:
            # 当前区域不在路径中，重新规划
            self.path = self.map_manager.find_path(current_area_id, self.exit_node_id)
            if len(self.path) > 0:
                self.path_index = 0
            else:
                return
        
        # 如果已经到达路径中的最后一个区域（出口），目标就是出口位置
        if self.path_index >= len(self.path) - 1:
            if self.map_manager is not None and self.exit_node_id in self.map_manager.areas:
                exit_area = self.map_manager.areas[self.exit_node_id]
                self.next_target_area_id = self.exit_node_id
                # 如果设置了出口位置，使用出口位置；否则使用出口区块中心
                if self.exit_pos is not None:
                    self.target_pos = self.exit_pos
                else:
                    self.target_pos = np.array(exit_area.center)
        else:
            # 目标指向路径中的下一个区域
            # 如果使用智能选择，在路口时重新选择
            if self.use_smart_choice and current_area.is_node():
                # 获取相邻区域
                adjacent_areas = self.map_manager.get_adjacent_areas(current_area_id)
                if len(adjacent_areas) > 0:
                    choices = [area.id for area in adjacent_areas if area.id in self.path or area.id == self.exit_node_id]
                    if len(choices) > 0:
                        # 获取出口位置
                        if self.exit_pos is not None:
                            exit_pos = self.exit_pos
                        elif self.exit_node_id in self.map_manager.areas:
                            exit_area = self.map_manager.areas[self.exit_node_id]
                            exit_pos = np.array(exit_area.center)
                        else:
                            exit_pos = np.array([0, 0])
                        
                        chosen_id = self.choice_smart(exit_pos, choices, neighbors or [], dangers or [], obstacles or [])
                        if chosen_id is not None:
                            # 更新路径，从当前区域到选择的下一个区域
                            try:
                                chosen_idx = self.path.index(chosen_id)
                                if chosen_idx > self.path_index:
                                    self.path_index = chosen_idx - 1
                            except ValueError:
                                pass
            
            next_area_id = self.path[self.path_index + 1] if self.path_index + 1 < len(self.path) else self.exit_node_id
            if self.map_manager is not None and next_area_id in self.map_manager.areas:
                next_area = self.map_manager.areas[next_area_id]
                self.next_target_area_id = next_area_id
                self.target_pos = np.array(next_area.center)
    
    def update_state(self, target, neighbors, obstacles, dangers, bound, current_time=0.0):
        # 如果已疏散，直接返回
        if self.evacuated:
            return
        
        # 更新路径规划
        self.update_path(neighbors, dangers, obstacles)
        
        # 如果路径规划成功，使用规划的目标点
        if self.target_pos is not None:
            target = self.target_pos
        
        # To update panic (优化：使用平方距离避免开方)
        perceive_radius_sq = self.perceive_radius ** 2
        neighbor_count = sum(1 for n in neighbors if np.dot(self.pos - n.pos, self.pos - n.pos) < perceive_radius_sq)
        danger = 0.0
        for each in dangers:
            danger = max(danger, each.danger(self.pos))
        self.panic = np.tanh(neighbor_count / np.pi + danger)

        # 生成新的正态分布概率因子（每次更新都不同）
        self.panic_probability_factor = np.random.normal(constants.PANIC_PROBABILITY_MEAN, constants.PANIC_PROBABILITY_STD)
        # 确保因子为正数（避免负值导致异常行为）
        self.panic_probability_factor = max(0.1, self.panic_probability_factor)

        # Updating velocity
        acc = self.calc_acc(target, neighbors, obstacles, dangers, bound)
        self.vel += acc * constants.SIMULATION_DT
        # In case one runs too fast or out of bound
        self.vel = self.vel / np.linalg.norm(self.vel) * constants.MAX_SPEED if np.linalg.norm(self.vel) > constants.MAX_SPEED else self.vel

        # Move
        self.pos += self.vel * constants.SIMULATION_DT
        
        # 检查是否在不可通行区域内，如果是则强制推回
        if self.map_manager is not None:
            current_area = self.map_manager.get_area_containing_point(self.pos)
            if current_area is not None and not self.map_manager.is_area_passable(current_area.id):
                nearest_area = self.map_manager.find_nearest_passable_area(self.pos)
                if nearest_area is not None:
                    # 将位置推回最近的可通行区域
                    push_dir = np.array(nearest_area.center) - self.pos
                    push_dist = float(np.linalg.norm(push_dir))
                    if push_dist > 1e-3:
                        push_dir = push_dir / push_dist
                        self.pos += push_dir * min(0.2, push_dist * 0.5)
        
        # 检查是否到达撤离点（疏散完成）
        # 只有当行人真正到达出口位置（exit_pos）时才标记为已疏散
        if self.exit_pos is not None and isinstance(self.exit_pos, np.ndarray):
            dist_to_exit = np.linalg.norm(self.pos - self.exit_pos)
            # 距离撤离点小于0.5m时认为已到达并疏散
            if dist_to_exit < 0.5:
                self.evacuated = True
                return
        
        # 检查是否到达路径中的中间目标（非出口）
        if self.target_pos is not None and isinstance(self.target_pos, np.ndarray):
            dist_to_target = np.linalg.norm(self.pos - self.target_pos)
            # 检查是否真正进入了目标区域
            target_reached = False
            if self.next_target_area_id != -1 and self.map_manager is not None:
                if self.next_target_area_id in self.map_manager.areas:
                    target_area = self.map_manager.areas[self.next_target_area_id]
                    if target_area.inside(self.pos):
                        target_reached = True
            
            # 如果到达的是出口区块，目标改为出口位置
            if self.next_target_area_id == self.exit_node_id:
                if self.exit_pos is not None:
                    self.target_pos = self.exit_pos
                # 不在这里标记疏散，需要真正到达exit_pos
            elif target_reached or dist_to_target < 0.3:  # 距离目标小于0.3m或已在目标区域内认为到达
                # 继续移动到路径中的下一个区域
                if len(self.path) > 0 and self.path_index < len(self.path) - 1:
                    self.path_index += 1
                    if self.path_index < len(self.path) and self.map_manager is not None:
                        next_area_id = self.path[self.path_index]
                        if next_area_id in self.map_manager.areas:
                            next_area = self.map_manager.areas[next_area_id]
                            self.next_target_area_id = next_area_id
                            # 如果下一个是出口，目标设为出口位置
                            if next_area_id == self.exit_node_id and self.exit_pos is not None:
                                self.target_pos = self.exit_pos
                            else:
                                self.target_pos = np.array(next_area.center)


# Area类已迁移到entities.AreaData
# 为了向后兼容，保留Area作为别名
Area = AreaData


class Map:
    """地图管理类"""
    def __init__(self):
        self.areas = {}  # {area_id: AreaData对象}
        self.next_area_id = 0
        return

    def add_road(self, lb, rt, node1_id=None, node2_id=None):
        """创建道路（property=0），连接两个路口"""
        area_id = self.next_area_id
        self.next_area_id += 1
        road = AreaData(lb, rt, area_id, property=0)
        self.areas[area_id] = road
        
        # 建立连接关系
        if node1_id is not None and node1_id in self.areas:
            if area_id not in self.areas[node1_id].adjacent:
                self.areas[node1_id].adjacent.append(area_id)
            if node1_id not in road.adjacent:
                road.adjacent.append(node1_id)
        if node2_id is not None and node2_id in self.areas:
            if area_id not in self.areas[node2_id].adjacent:
                self.areas[node2_id].adjacent.append(area_id)
            if node2_id not in road.adjacent:
                road.adjacent.append(node2_id)
        
        return area_id
    
    def add_node(self, lb, rt):
        """创建路口（property=1）"""
        area_id = self.next_area_id
        self.next_area_id += 1
        node = AreaData(lb, rt, area_id, property=1)
        self.areas[area_id] = node
        return area_id
    
    def add_forbidden_area(self, lb, rt):
        """创建不可进入区域，所有边界设为jammed"""
        area_id = self.next_area_id
        self.next_area_id += 1
        forbidden = AreaData(lb, rt, area_id, property=0)
        # 设置所有边界为jammed
        forbidden.jammed = ['left', 'right', 'upper', 'lower']
        self.areas[area_id] = forbidden
        return area_id
    
    def get_area_containing_point(self, pos):
        """查找包含指定点的区域（优化：缓存最近查找的区域）"""
        # 简单的缓存优化：如果点在缓存区域内，直接返回
        if hasattr(self, '_cached_area') and self._cached_area is not None:
            if self._cached_area.inside(pos):
                return self._cached_area
        
        # 遍历查找
        for area in self.areas.values():
            if area.inside(pos):
                self._cached_area = area  # 缓存结果
                return area
        self._cached_area = None
        return None
    
    def get_adjacent_areas(self, area_id):
        """获取指定区域的相邻区域列表"""
        if area_id not in self.areas:
            return []
        adjacent_ids = self.areas[area_id].adjacent
        return [self.areas[aid] for aid in adjacent_ids if aid in self.areas]
    
    def connect_areas(self, area1_id, area2_id):
        """建立两个区域的相邻关系"""
        if area1_id in self.areas and area2_id in self.areas:
            if area2_id not in self.areas[area1_id].adjacent:
                self.areas[area1_id].adjacent.append(area2_id)
            if area1_id not in self.areas[area2_id].adjacent:
                self.areas[area2_id].adjacent.append(area1_id)
    
    def is_area_passable(self, area_id):
        """判断区域是否可通行（不是不可进入区域）"""
        if area_id not in self.areas:
            return False
        area = self.areas[area_id]
        # 如果所有边界都被jammed，则是不可进入区域
        return len(area.jammed) < 4
    
    def find_path(self, start_area_id, end_area_id):
        """使用A*算法查找从起点到终点的路径（已迁移到PathPlanner）"""
        # 为了向后兼容，保留此方法
        # 实际实现已迁移到managers.PathPlanner
        from .managers import PathPlanner
        planner = PathPlanner(self)
        return planner.find_path(start_area_id, end_area_id)
    
    def get_passable_areas(self):
        """获取所有可通行的区域"""
        return [area_id for area_id, area in self.areas.items() if self.is_area_passable(area_id)]
    
    def find_nearest_passable_area(self, pos):
        """查找距离指定位置最近的可通行区域"""
        min_dist = float('inf')
        nearest_area = None
        for area_id in self.get_passable_areas():
            area = self.areas[area_id]
            dist = np.linalg.norm(pos - np.array(area.center))
            if dist < min_dist:
                min_dist = dist
                nearest_area = area
        return nearest_area
