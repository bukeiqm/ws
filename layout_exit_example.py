"""Layout数组出口示例：演示如何在layout数组中指定出口（值为3）"""

import numpy as np
from pedestrian_evacuation import Map

# ========== 示例：在layout数组中指定出口 ==========
print("示例：在layout数组中指定出口（3=出口路口）")

# 创建一个简单的地图布局
# 2 = 路口（property=1）
# 1 = 道路（property=0）
# 0 = 不可通行区域
# 3 = 出口（同时也是路口）
layout = [
    [0, 0, 0, 0, 0, 0],
    [0, 2, 1, 1, 3, 0],  # 路口-道路-道路-出口
    [0, 1, 0, 0, 1, 0],  # 道路-不可通行-不可通行-道路
    [0, 1, 0, 0, 1, 0],  # 道路-不可通行-不可通行-道路
    [0, 3, 1, 1, 2, 0],  # 出口-道路-道路-路口
    [0, 0, 0, 0, 0, 0],
]

# 创建地图
map_manager = Map()

cell_size = 2.0  # 每个网格的大小（米）
grid_rows = len(layout)
grid_cols = len(layout[0]) if layout else 0

# 检查layout中是否有值为2或3的区域（路口或出口）
has_nodes = any(2 in row or 3 in row for row in layout)

passable_areas = []
area_grid = {}
node_grid = {}
exits = []  # 存储出口信息

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
                    print(f"创建出口: 位置({i},{j}), ID={area_id}, 位置={exit_pos}")
    
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
                print(f"创建道路: 位置({i},{j}), ID={area_id}, 连接路口={node1_id},{node2_id}")
    
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

print(f"\n总共创建了 {len(map_manager.areas)} 个区域")
print(f"可通行区域: {len(passable_areas)} 个")
print(f"路口数量: {len(node_grid)} 个")
print(f"出口数量: {len(exits)} 个")

# 显示出口信息
print("\n出口信息:")
for exit_id, exit_pos in exits:
    print(f"  出口ID: {exit_id}, 位置: ({exit_pos[0]:.1f}, {exit_pos[1]:.1f})")

print("\n✓ 示例运行成功！")
print("\n说明:")
print("- 2 = 路口（property=1），用于路径规划的关键节点")
print("- 1 = 道路（property=0），连接路口的通道")
print("- 0 = 不可通行区域（forbidden area）")
print("- 3 = 出口（同时也是路口），行人疏散的目标位置")
