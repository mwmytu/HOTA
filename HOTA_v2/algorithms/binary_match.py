"""
二分图分配算法
"""
import networkx as nx
from utils import *
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from networkx.algorithms import bipartite


# 构建二分图
def build_bipartite_graph(workers, tasks, current_time):
    # 创建一个空二分图
    g = nx.Graph()

    # 增加工人顶点
    for worker in workers:
        remain_time = add_date(worker.end_time) - current_time  # 计算工人的剩余时间
        g.add_node("worker_" + str(worker.id), bipartite=0, attributes=worker, remain_time=remain_time)

    # 增加任务顶点
    for task in tasks:
        if task.task_type in ['TypeA', 'TypeB']:
            remain_allocate = 1
            remain_time = add_date(task.start_time) + timedelta(
                hours=(len(task.assigned_workers) + 1) * task.interval) - current_time
        else:
            remain_allocate = task.num_area - len(task.assigned_workers)
            remain_time = add_date(task.end_time) - current_time
        g.add_node("task_" + str(task.id), bipartite=1, attributes=task,
                   task_type=task.task_type, remain_allocate=remain_allocate,
                   remain_time=remain_time)

    # 添加边
    for task in tasks:
        for worker in workers:
            if not worker.is_available(current_time):  # 如果工人当前不可用，跳过此工人
                continue

            if task.task_type in ['TypeA', 'TypeB']:
                task_start = add_date(task.start_time) + timedelta(hours=len(task.assigned_workers) * task.interval)
                worker_start = calculate_worker_start_time(worker, current_time)
                start_task = worker_start if worker_start > task_start else task_start
            else:
                start_task = calculate_worker_start_time(worker, current_time)

            # 计算到达并完成任务需要的时间
            travel_time = calculate_travel_time(worker, task, current_time)
            sensing_time = calculate_pair_sensing_time(worker, task)
            total_time = travel_time + sensing_time

            # 边权重
            # weight_reward = calculate_pair_sensing_bid(worker, task)
            weight_reward = calculate_pair_sensing_reward(worker, task, current_time)
            weight_quality = calculate_pair_sensing_quality(worker, task)

            # 检查工人完成任务的时间是否在工人的工作时间和任务截止时间之内
            expected_finish_time = start_task + timedelta(hours=total_time)
            if task.task_type in ['TypeA', 'TypeB']:
                if (expected_finish_time <= add_date(worker.end_time) and
                        expected_finish_time <= (
                                add_date(task.start_time) + timedelta(hours=len(task.assigned_workers) + 1))):
                    if weight_quality >= task.quality_threshold:
                        g.add_edge("worker_" + str(worker.id), "task_" + str(task.id),
                                   weight_reward=weight_reward, weight_quality=weight_quality)
            else:
                if (expected_finish_time <= add_date(worker.end_time) and
                        expected_finish_time <= add_date(task.end_time)):
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


def calculate_remaining_time(task, current_time):
    if task.task_type in ['TypeA', 'TypeB']:
        remaining_time = (add_date(task.start_time) + timedelta(hours=(len(task.assigned_workers) + 1) * task.interval) - current_time).seconds
    else:
        remaining_time = (add_date(task.end_time) - current_time).seconds
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
                task_start = add_date(task.start_time) + timedelta(hours=task.interval * len(task.assigned_workers))
                worker_start = calculate_worker_start_time(worker, current_time)
                start_task = worker_start if worker_start > task_start else task_start
            else:
                start_task = calculate_worker_start_time(worker, current_time)

            # 计算需要的时间
            travel_time = calculate_travel_time(worker, task, current_time)
            if travel_time > 2 * time_slot or (remaining_hours > 4 and travel_time > time_slot):
                continue
            sensing_time = calculate_pair_sensing_time(worker, task)
            total_time = travel_time + sensing_time
            # weight_reward = calculate_pair_sensing_bid(worker, task)
            weight_reward = calculate_pair_sensing_reward(worker, task, current_time)
            weight_quality = calculate_pair_sensing_quality(worker, task)

            # 检查任务完成时间
            expected_finish_time = start_task + timedelta(hours=total_time)
            if task.task_type in ['TypeA', 'TypeB']:
                if (expected_finish_time <= add_date(worker.end_time) and
                        expected_finish_time <= (add_date(task.start_time) +
                                                 timedelta(hours=task.interval * (len(task.assigned_workers) + 1)))):
                    if weight_quality >= task.quality_threshold:
                        graph.add_edge("worker_" + str(worker.id), "task_" + str(task.id), capacity=1,
                                       weight_reward=weight_reward, weight_quality=weight_quality)
            else:
                if (expected_finish_time <= add_date(worker.end_time) and
                        expected_finish_time <= add_date(task.end_time)):
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
