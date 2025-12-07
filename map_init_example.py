"""地图初始化示例：演示如何在初始化时指定所有道路和路口"""

import numpy as np
from pedestrian_evacuation import Map

# ========== 示例1：使用列表格式 ==========
print("示例1：使用列表格式初始化地图")

# 定义路口（左下角和右上角坐标）
nodes = [
    ([0, 0], [2, 2]),      # 路口0：左下(0,0) 右上(2,2)
    ([4, 0], [6, 2]),      # 路口1：左下(4,0) 右上(6,2)
    ([0, 4], [2, 6]),      # 路口2：左下(0,4) 右上(2,6)
    ([4, 4], [6, 6]),      # 路口3：左下(4,4) 右上(6,6)
]

# 定义道路（连接路口）
# 格式：(lb, rt, node1_idx, node2_idx)
# node1_idx和node2_idx是nodes列表中的索引
roads = [
    ([2, 0], [4, 2], 0, 1),    # 道路：连接路口0和路口1
    ([0, 2], [2, 4], 0, 2),    # 道路：连接路口0和路口2
    ([2, 4], [4, 6], 2, 3),    # 道路：连接路口2和路口3
    ([4, 2], [6, 4], 1, 3),    # 道路：连接路口1和路口3
]

# 创建地图
map1 = Map(nodes=nodes, roads=roads)
print(f"创建了 {len(map1.areas)} 个区域")
print(f"路口ID列表: {map1.get_node_ids()}")

# ========== 示例2：使用字典格式 ==========
print("\n示例2：使用字典格式初始化地图")

nodes_dict = [
    {"lb": [0, 0], "rt": [2, 2]},
    {"lb": [4, 0], "rt": [6, 2]},
    {"lb": [0, 4], "rt": [2, 6]},
    {"lb": [4, 4], "rt": [6, 6]},
]

roads_dict = [
    {"lb": [2, 0], "rt": [4, 2], "node1_idx": 0, "node2_idx": 1},
    {"lb": [0, 2], "rt": [2, 4], "node1_idx": 0, "node2_idx": 2},
    {"lb": [2, 4], "rt": [4, 6], "node1_idx": 2, "node2_idx": 3},
    {"lb": [4, 2], "rt": [6, 4], "node1_idx": 1, "node2_idx": 3},
]

map2 = Map(nodes=nodes_dict, roads=roads_dict)
print(f"创建了 {len(map2.areas)} 个区域")
print(f"路口ID列表: {map2.get_node_ids()}")

# ========== 示例3：只创建路口，后续手动添加道路 ==========
print("\n示例3：只创建路口，后续手动添加道路")

map3 = Map(nodes=nodes)
node_ids = map3.get_node_ids()
print(f"路口ID: {node_ids}")

# 手动添加道路，使用路口ID
map3.add_road([2, 0], [4, 2], node_ids[0], node_ids[1])
map3.add_road([0, 2], [2, 4], node_ids[0], node_ids[2])
print(f"手动添加道路后，共有 {len(map3.areas)} 个区域")

# ========== 示例4：混合使用（部分道路连接，部分不连接） ==========
print("\n示例4：混合使用（部分道路连接路口，部分不连接）")

# 创建一些独立的路口
nodes_mixed = [
    ([0, 0], [2, 2]),
    ([4, 0], [6, 2]),
]

# 创建道路：有些连接路口，有些不连接
roads_mixed = [
    ([2, 0], [4, 2], 0, 1),      # 连接路口0和1
    ([8, 0], [10, 2], None, None),  # 独立道路，不连接路口
]

map4 = Map(nodes=nodes_mixed, roads=roads_mixed)
print(f"创建了 {len(map4.areas)} 个区域")
print(f"路口ID: {map4.get_node_ids()}")

# 验证连接关系
for node_id in map4.get_node_ids():
    node = map4.areas[node_id]
    adjacent_count = len(node.adjacent)
    print(f"路口 {node_id} 连接了 {adjacent_count} 个区域")

print("\n✓ 所有示例运行成功！")
