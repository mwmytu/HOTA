"""
二分图
"""
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from utils import *
from datetime import timedelta
from scipy.optimize import linear_sum_assignment
from networkx.algorithms import bipartite


# 构建二分图
def build_bipartite_graph(workers, tasks, current_time):
    # 创建一个空二分图
    g = nx.Graph()

    # 增加工人顶点
    for worker in workers:
        remain_time = to_dt(worker.end_time) - current_time  # 计算工人的剩余时间
        g.add_node("worker_" + str(worker.id), bipartite=0, attributes=worker, remain_time=remain_time)

    # 增加任务顶点
    for task in tasks:
        if task.task_type in ['TypeA', 'TypeB']:
            remain_allocate = 1
            remain_time = to_dt(task.start_time) + timedelta(
                hours=(len(task.assigned_workers) + 1) * task.interval) - current_time
        else:
            remain_allocate = task.num_area - len(task.assigned_workers)
            remain_time = to_dt(task.end_time) - current_time
        g.add_node("task_" + str(task.id), bipartite=1, attributes=task,
                   task_type=task.task_type, remain_allocate=remain_allocate,
                   remain_time=remain_time)

    # 添加边
    for task in tasks:
        for worker in workers:
            if not worker.is_available(current_time):  # 如果工人当前不可用，跳过此工人
                continue

            if task.task_type in ['TypeA', 'TypeB']:
                task_start = to_dt(task.start_time) + timedelta(hours=len(task.assigned_workers) * task.interval)
                worker_start = calculate_worker_start_time(worker, current_time)
                start_task = worker_start if worker_start > task_start else task_start
            else:
                start_task = calculate_worker_start_time(worker, current_time)

            # 计算到达并完成任务需要的时间
            travel_time = calculate_travel_time(worker, task)
            sensing_time = calculate_pair_sensing_time(worker, task)
            total_time = travel_time + sensing_time

            # 边权重
            weight_reward = calculate_pair_sensing_bid(worker, task)
            weight_quality = calculate_pair_sensing_quality(worker, task)

            # 检查工人完成任务的时间是否在工人的工作时间和任务截止时间之内
            expected_finish_time = start_task + timedelta(hours=total_time)
            if task.task_type in ['TypeA', 'TypeB']:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= (
                                to_dt(task.start_time) + timedelta(hours=len(task.assigned_workers) + 1))):
                    if weight_quality >= task.quality_threshold:
                        g.add_edge("worker_" + str(worker.id), "task_" + str(task.id),
                                   weight_reward=weight_reward, weight_quality=weight_quality)
            else:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= to_dt(task.end_time)):
                    if weight_quality >= task.quality_threshold:
                        g.add_edge("worker_" + str(worker.id), "task_" + str(task.id),
                                   weight_reward=weight_reward, weight_quality=weight_quality)

    return g


# 对二分图做可视化
def visualize_graph(g):
    # 创建一个图形和轴
    plt.figure(figsize=(12, 8))

    # 绘制图的布局
    pos = nx.bipartite_layout(g, nodes=[n for n, d in g.nodes(data=True) if d['bipartite'] == 0])

    # 绘制节点
    nx.draw_networkx_nodes(g, pos, nodelist=[n for n, d in g.nodes(data=True) if d['bipartite'] == 0],
                           node_color='lightblue', node_size=500, label='Workers')
    nx.draw_networkx_nodes(g, pos, nodelist=[n for n, d in g.nodes(data=True) if d['bipartite'] == 1],
                           node_color='lightgreen', node_size=500, label='Tasks')

    # 绘制边
    nx.draw_networkx_edges(g, pos)

    # 绘制标签
    nx.draw_networkx_labels(g, pos, labels={n: n for n in g.nodes()}, font_size=10, font_family='sans-serif')

    # 添加图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Workers'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Tasks')
    ]
    plt.legend(handles=legend_elements, loc='best')

    # 显示图形
    plt.title('Bipartite Graph of Workers and Tasks')
    plt.show()


def binary_match_algorithm(workers, tasks, current_time):
    g = build_bipartite_graph(workers, tasks, current_time)
    for u, v, data in g.edges(data=True):
        # data['weight'] = -data['weight_reward']  # 取负，最大化weight_reward即最小化原始weight_reward
        task_remain_time = g.nodes[v]['remain_time']  # 获取任务的剩余时间
        if task_remain_time <= timedelta(hours=2):  # 剩余时间小于1小时
            data['weight'] = -data['weight_reward'] + 20  # 提升其权重，使其优先分配
            # print(f"新加了权重，值为：{data['weight']}")
        else:
            data['weight'] = -data['weight_reward']  # 取负以最大化

    # 使用匈牙利算法进行最大匹配
    matching = bipartite.maximum_matching(g, top_nodes={n for n, d in g.nodes(data=True) if d['bipartite'] == 0})
    print(f"匈牙利算法最大匹配的结果：{matching}")

    # 准备任务分配的结果
    allocations = {task.id: [] for task in tasks}  # 初始化所有任务的分配列表

    task_allocation_count = {f"task_{task.id}": 0 for task in tasks}  # 记录每个任务已分配的工人数量

    # 迭代匹配结果，确定有效的分配
    for worker, task in matching.items():
        if worker.startswith("worker_") and task.startswith("task_"):
            task_id = int(task.split("_")[1])  # 提取任务 ID
            if task_allocation_count[task] < g.nodes[task]['remain_allocate']:
                worker_id = int(worker.split("_")[1])
                allocations[task_id].append(worker_id)  # 将工人添加到任务的分配列表
                task_allocation_count[task] += 1

    return allocations


def calculate_remaining_time(task, current_time):
    if task.task_type in ['TypeA', 'TypeB']:
        remaining_time = (to_dt(task.start_time) + timedelta(hours=(len(task.assigned_workers) + 1) * task.interval) - current_time).seconds
    else:
        remaining_time = (to_dt(task.end_time) - current_time).seconds
    remaining_hours = remaining_time / 3600
    return remaining_hours


# 创建最大流图
def build_max_flow_graph(workers, tasks, current_time, time_slot):
    time_slot = time_slot / 60
    # 创建一个空的有向图
    graph = nx.DiGraph()

    # 添加源和汇节点
    source = 'source'
    sink = 'sink'
    graph.add_node(source)
    graph.add_node(sink)

    # 添加工人节点和任务节点
    for worker in workers:
        graph.add_node("worker_" + str(worker.id))

        # 从工人节点到汇节点的边
        graph.add_edge(source, "worker_" + str(worker.id), capacity=1)

    # sorted_tasks = sorted(tasks, key=lambda task: calculate_remaining_time(task, current_time))
    sorted_tasks = sorted(tasks, key=lambda task: (calculate_remaining_time(task, current_time), -task.complexity))
    for task in sorted_tasks:
        remain_allocate = 1 if task.task_type in ['TypeA', 'TypeB'] else task.num_area - len(task.assigned_workers)
        graph.add_node("task_" + str(task.id))

        # 从源节点到任务节点的边
        graph.add_edge("task_" + str(task.id), sink, capacity=remain_allocate)

    # 添加工人与任务之间的边
    for task in tasks:
        # 计算任务的剩余时间
        remaining_time = calculate_remaining_time(task, current_time)
        remaining_hours = remaining_time / 3600

        for worker in workers:
            if not worker.is_available(current_time):
                continue

            # 计算任务开始时间和工人开始时间
            if task.task_type in ['TypeA', 'TypeB']:
                task_start = to_dt(task.start_time) + timedelta(hours=task.interval * len(task.assigned_workers))
                worker_start = calculate_worker_start_time(worker, current_time)
                start_task = worker_start if worker_start > task_start else task_start
            else:
                start_task = calculate_worker_start_time(worker, current_time)

            # 计算需要的时间
            travel_time = calculate_travel_time(worker, task)
            if travel_time > 2 * time_slot or (remaining_hours > 4 and travel_time > time_slot):
                continue
            sensing_time = calculate_pair_sensing_time(worker, task)
            total_time = travel_time + sensing_time
            weight_reward = calculate_pair_sensing_bid(worker, task)
            weight_quality = calculate_pair_sensing_quality(worker, task)

            # 检查任务完成时间
            expected_finish_time = start_task + timedelta(hours=total_time)
            if task.task_type in ['TypeA', 'TypeB']:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= (to_dt(task.start_time) +
                                                 timedelta(hours=task.interval * (len(task.assigned_workers) + 1)))):
                    if weight_quality >= task.quality_threshold:
                        graph.add_edge("worker_" + str(worker.id), "task_" + str(task.id), capacity=1,
                                       weight_reward=weight_reward, weight_quality=weight_quality)
            else:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= to_dt(task.end_time)):
                    if weight_quality >= task.quality_threshold:
                        graph.add_edge("worker_" + str(worker.id), "task_" + str(task.id), capacity=1,
                                       weight_reward=weight_reward, weight_quality=weight_quality)

    return graph


def maximum_flow_allocation(workers, tasks, current_time, time_slot):
    # 构建最大流图
    max_flow_graph = build_max_flow_graph(workers, tasks, current_time, time_slot)

    # 使用最大流算法
    flow_value, flow_dict = nx.maximum_flow(max_flow_graph, _s='source', _t='sink')
    # print(f"最大流为：{flow_value}，分配结果为：{flow_dict}")
    # 使用最小费用最大流算法
    # flow_cost, flow_dict = nx.network_simplex(max_flow_graph)
    # print(f"成本为：{flow_cost}，分配结果为：{flow_dict}")

    # # 可视化图
    # pos = nx.spring_layout(max_flow_graph)  # 选择布局
    # nx.draw(max_flow_graph, pos, with_labels=True, node_color='lightblue', arrows=True)
    #
    # # 显示边的容量和流量
    # edge_labels = {(u, v): f"{flow_dict[u][v]}/{max_flow_graph[u][v]['capacity']}" for u, v in max_flow_graph.edges()}
    # nx.draw_networkx_edge_labels(max_flow_graph, pos, edge_labels=edge_labels)
    #
    # plt.title(f'Max Flow: {flow_value}')
    # plt.show()

    # 构建结果字典
    assignments = {task.id: [] for task in tasks}

    for worker in workers:
        worker_node = f"worker_{worker.id}"
        for task in tasks:
            task_node = f"task_{task.id}"
            if flow_dict[worker_node].get(task_node, 0) > 0:
                assignments[task.id].append(worker.id)
                break

    return assignments


# 最大分配，考虑双权重
def binary_match_algorithm_cost_matrix(workers, tasks, current_time):
    graph = build_bipartite_graph(workers, tasks, current_time)
    # 获取工人和任务的节点列表
    worker_nodes = [node for node in graph.nodes if graph.nodes[node]['bipartite'] == 0]
    # print(worker_nodes)
    task_nodes = [node for node in graph.nodes if graph.nodes[node]['bipartite'] == 1]
    # print(task_nodes)

    # 初始化成本矩阵
    cost_matrix = np.full((len(worker_nodes), len(task_nodes)), np.inf)

    worker_indices = {n: i for i, n in enumerate(graph.nodes) if graph.nodes[n]['bipartite'] == 0}
    task_indices = {n: j for j, n in enumerate(graph.nodes) if graph.nodes[n]['bipartite'] == 1}

    # 填充成本矩阵
    for worker in graph.nodes:
        if graph.nodes[worker]['bipartite'] == 0:  # 是工人
            for task in graph.neighbors(worker):
                if graph.nodes[task]['bipartite'] == 1:  # 是任务
                    weight_reward = graph[worker][task]['weight_reward']
                    weight_quality = graph[worker][task]['weight_quality']

                    # 设定成本为 reward - quality 组合
                    cost_matrix[worker_indices[worker]][task_indices[task]] = weight_reward - weight_quality

    # 使用匈牙利算法找到最小成本匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 准备分配结果
    assignment = {}
    for worker_idx, task_idx in zip(row_ind, col_ind):
        if cost_matrix[worker_idx][task_idx] < float('inf'):
            worker_node = list(worker_indices.keys())[worker_idx]
            task_node = list(task_indices.keys())[task_idx]

            if task_node not in assignment:
                assignment[task_node] = []
            assignment[task_node].append(worker_node)

            # 检查任务的 remain_allocate 限制
            if len(assignment[task_node]) >= graph.nodes[task_node]['remain_allocate']:
                break

    return assignment


# 按报酬从小到大分配
def binary_match_greedy_reward(workers, tasks, current_time):
    # 构建二分图
    G = build_bipartite_graph(workers, tasks, current_time)

    # 结果字典，任务ID映射到工人ID列表
    task_assignments = {task.id: [] for task in tasks}

    # 用于跟踪已分配的工人ID
    assigned_worker_ids = set()

    # 遍历每个任务，按边权重排序工人
    for task in tasks:
        task_id = "task_" + str(task.id)
        # 从任务节点获取remain_allocate属性
        remain_allocate = G.nodes[task_id].get('remain_allocate', 0)

        # 获取与任务相连的工人及其权重
        worker_weights = []
        for worker in G.neighbors(task_id):
            if worker.startswith("worker_"):
                worker_id = int(worker.split("_")[1])
                # 如果工人已被分配，则跳过
                if worker_id in assigned_worker_ids:
                    continue
                # 获取边的权重
                edge_data = G[worker][task_id]
                weight_reward = edge_data['weight_reward']
                weight_quality = edge_data['weight_quality']
                worker_weights.append((worker_id, weight_reward / weight_quality))

        # 根据权重对工人进行排序
        worker_weights.sort(key=lambda x: x[1])  # 按权重升序排序

        # 选择权重最小的工人，直到达到remain_allocate限制
        for worker_id, weight in worker_weights:
            if len(task_assignments[task.id]) < remain_allocate:
                task_assignments[task.id].append(worker_id)
                assigned_worker_ids.add(worker_id)  # 记录已分配的工人

    return task_assignments


def assign_workers_to_tasks(workers, tasks, current_time):
    graph = build_bipartite_graph(workers, tasks, current_time)
    visualize_graph(graph)
    # 创建一个新的图用于流量优化
    flow_graph = nx.DiGraph()

    source = 'source'
    sink = 'sink'

    # 添加源节点和汇节点
    flow_graph.add_node(source)
    flow_graph.add_node(sink)

    workers_assignments = {}  # 用于存储每个任务的工人分配情况

    # 添加工人到源节点的边
    for worker in graph.nodes():
        if graph.nodes[worker]['bipartite'] == 0:  # Worker nodes
            flow_graph.add_edge(source, worker, capacity=1)  # 每个工人只能分配一次

    # 添加任务到汇节点的边
    for task in graph.nodes():
        if graph.nodes[task]['bipartite'] == 1:  # Task nodes
            remain_allocate = graph.nodes[task]['remain_allocate']
            flow_graph.add_edge(task, sink, capacity=remain_allocate)  # 根据任务需求分配能力
            workers_assignments[task] = []  # 初始化每个任务的工人分配列表

    # 添加工人与任务之间的边，并设置权重
    for u, v, data in graph.edges(data=True):
        weight_reward = data['weight_reward']
        weight_quality = data['weight_quality']

        # 我们可以将权重合并成一个总权重（可以根据实际情况进行调整）
        total_weight = weight_reward / weight_quality  # 这里假设我们在最小化报酬同时最大化质量

        flow_graph.add_edge(u, v, capacity=1, weight=total_weight)  # 每个边的容量为1

    # 计算最小费用流
    flow_cost, flow_dict = nx.network_simplex(flow_graph)

    # 提取分配结果
    for worker in graph.nodes():
        if graph.nodes[worker]['bipartite'] == 0:  # Worker nodes
            for task in graph.neighbors(worker):
                if flow_dict[worker][task] > 0:  # 如果分配了工人给任务
                    workers_assignments[task].append(worker)

    return workers_assignments, flow_cost  # 返回任务ID为键，工人ID列表为值的分配结果和总成本
