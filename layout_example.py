"""Layout数组示例：演示如何使用2=路口，1=道路，0=不可通行区域"""

import numpy as np
from pedestrian_evacuation import Map

# ========== 示例：使用新的layout格式 ==========
print("示例：使用layout数组创建地图（2=路口，1=道路，0=不可通行）")

# 创建一个简单的地图布局
# 2 = 路口（property=1）
# 1 = 道路（property=0）
# 0 = 不可通行区域
layout = [
    [0, 0, 0, 0, 0, 0],
    [0, 2, 1, 1, 2, 0],  # 路口-道路-道路-路口
    [0, 1, 0, 0, 1, 0],  # 道路-不可通行-不可通行-道路
    [0, 1, 0, 0, 1, 0],  # 道路-不可通行-不可通行-道路
    [0, 2, 1, 1, 2, 0],  # 路口-道路-道路-路口
    [0, 0, 0, 0, 0, 0],
]

# 创建地图
map_manager = Map()

cell_size = 2.0  # 每个网格的大小（米）
grid_rows = len(layout)
grid_cols = len(layout[0]) if layout else 0

# 检查layout中是否有值为2的区域（路口）
has_nodes = any(2 in row for row in layout)

passable_areas = []
area_grid = {}
node_grid = {}

if has_nodes:
    # 使用新模式：2=路口，1=道路，0=不可通行
    
    # 第一步：创建所有路口（值为2的区域）
    for j in range(grid_rows):
        for i in range(grid_cols):
            if layout[j][i] == 2:
                lb = [i * cell_size, j * cell_size]
                rt = [(i + 1) * cell_size, (j + 1) * cell_size]
                area_id = map_manager.add_node(lb, rt)
                passable_areas.append(area_id)
                area_grid[(i, j)] = area_id
                node_grid[(i, j)] = area_id
                print(f"创建路口: 位置({i},{j}), ID={area_id}")
    
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

# 验证路口和道路的区别
print("\n验证区域类型:")
for node_pos, node_id in node_grid.items():
    node = map_manager.areas[node_id]
    print(f"路口 {node_id} (位置{node_pos}): property={node.property}, 相邻区域数={len(node.adjacent)}")

print("\n✓ 示例运行成功！")
print("\n说明:")
print("- 2 = 路口（property=1），用于路径规划的关键节点")
print("- 1 = 道路（property=0），连接路口的通道")
print("- 0 = 不可通行区域（forbidden area）")
