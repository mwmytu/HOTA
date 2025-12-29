"""
考虑最大化质量yi最小化报酬，实现pareto最优
"""
from typing import List

import networkx as nx

# from algorithms.greedy import greedy_bid_assign
from utils import *


def build_graph(workers, tasks, current_time, time_slot):
    # 创建一个空二分图
    g = nx.Graph()

    # 增加工人顶点，不考虑可用时间，先计算，先不用，后期可以直接在代码上改
    for worker in workers:
        remain_time = to_dt(worker.end_time) - current_time  # 计算工人的剩余时间
        g.add_node("worker_" + str(worker.id), bipartite=0, attributes=worker, remain_time=remain_time)

    # 增加任务顶点
    for task in tasks:
        if task.task_type in ['TypeA', 'TypeB']:
            remain_time = (to_dt(task.start_time) + timedelta(hours=task.interval * (len(task.assigned_workers) + 1))
                           - current_time)
            remain_allocate = 1
        else:
            remain_time = to_dt(task.end_time) - current_time
            remain_allocate = task.num_area - len(task.assigned_workers)
        g.add_node("task_" + str(task.id), bipartite=1, attributes=task,
                   remain_time=remain_time, remain_allocate=remain_allocate)

    # 添加边
    for task in tasks:
        for worker in workers:

            if task.task_type in ['TypeA', 'TypeB']:
                task_start = to_dt(task.start_time) + timedelta(hours=task.interval * len(task.assigned_workers))
                worker_start = calculate_worker_start_time(worker, current_time)
                start_task = worker_start if worker_start > task_start else task_start
            else:
                start_task = calculate_worker_start_time(worker, current_time)

            # 计算到达并完成任务需要的时间
            travel_time = calculate_travel_time(worker, task)
            # if travel_time > 2 * time_slot:  # 改成1个
            #     continue
            sensing_time = calculate_pair_sensing_time(worker, task)
            total_time = travel_time + sensing_time

            # 计算边的权重
            weight_start = start_task
            weight_sensing_time = sensing_time
            weight_travel_time = travel_time
            weight_distance = calculate_distance(worker, task)
            weight_reward = calculate_pair_sensing_bid(worker, task)
            weight_quality = calculate_pair_sensing_quality(worker, task)

            # 检查工人完成任务的时间是否在工人的工作时间和任务截止时间之内
            expected_finish_time = start_task + timedelta(hours=total_time)
            if task.task_type in ['TypeA', 'TypeB']:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= (to_dt(task.start_time) +
                                                 timedelta(hours=task.interval * (len(task.assigned_workers) + 1)))):
                    if weight_quality >= task.quality_threshold:
                        g.add_edge("worker_" + str(worker.id), "task_" + str(task.id),
                                   weight_start=weight_start, weight_sensing_time=weight_sensing_time,
                                   weight_travel_time=weight_travel_time, weight_distance=weight_distance,
                                   weight_reward=weight_reward,
                                   weight_quality=weight_quality)
            else:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= to_dt(task.end_time)):
                    if weight_quality >= task.quality_threshold:
                        g.add_edge("worker_" + str(worker.id), "task_" + str(task.id),
                                   weight_start=weight_start, weight_sensing_time=weight_sensing_time,
                                   weight_travel_time=weight_travel_time, weight_distance=weight_distance,
                                   weight_reward=weight_reward,
                                   weight_quality=weight_quality)

    return g


# 要改，因为是双目标
def create_individual_from_graph(graph):
    # 获取所有任务节点的ID
    task_ids = [int(node.split('_')[1]) for node in graph.nodes() if node.startswith("task_")]
    # 初始化个体
    individual = {task_id: [] for task_id in task_ids}

    # 记录已分配的工人
    assigned_workers = set()

    random.shuffle(task_ids)
    for task_id in task_ids:

        task_node = f"task_{task_id}"
        # if task_node not in graph.nodes:
        #     continue

        # 得到该任务所需工人数
        required_workers_count = graph.nodes[task_node]['remain_allocate']
        remain_time = graph.nodes[task_node]['remain_time']

        # # 找到所有与任务节点相连的工人节点
        # possible_workers = [(n, graph[task_node][n]['weight_reward']) for n in graph.neighbors(task_node) if
        #                     n.startswith("worker_")]
        # 找到所有与任务节点相连的工人节点
        possible_workers = []
        for n in graph.neighbors(task_node):
            if n.startswith("worker_"):
                # weight_reward = graph[task_node][n]['weight_reward']
                weight_quality = graph[task_node][n]['weight_quality']
                weight_travel_time = graph[task_node][n]['weight_travel_time']

                # 检查条件
                if remain_time > timedelta(hours=2) and weight_travel_time > 1:
                    continue  # 如果剩余时间大于2且工人边权重超过0.5，跳过该工人
                possible_workers.append((n, weight_quality))

        # 对工人节点进行随机排序，以确保分配的公平性
        random.shuffle(possible_workers)

        # 按 `weight_reward` 从小到大排序
        # possible_workers.sort(key=lambda x: x[1], reverse=True)

        for worker, weight in possible_workers:
            worker_id = int(worker.split("_")[1])  # 转换为整数
            if worker_id not in assigned_workers and len(individual[task_id]) < required_workers_count:
                individual[task_id].append(worker_id)
                assigned_workers.add(worker_id)

            # 如果已满足所需工人数，则跳出循环
            if len(individual[task_id]) >= required_workers_count:
                break
        else:
            individual[task_id] = []

    return individual


def initialize_population(population_size, graph):
    population = []
    # # 加入一个二分贪心解
    # assignments = greedy_bid_assign(workers, tasks, current_time, 30)
    # population.append(assignments)
    for _ in range(population_size):
        individual = create_individual_from_graph(graph)
        # print(individual)
        population.append(individual)
    return population


# 不适应度—报酬+质量-惩罚项
def calculate_unfitness(individual, graph):
    # print(f"个体为：{individual}")
    total_reward, total_quality = 0, 0

    # 获取所有任务
    task_ids = [int(node.split('_')[1]) for node in graph.nodes() if node.startswith("task_")]

    for task_id in task_ids:
        task_node = f"task_{task_id}"
        # 获取任务的剩余时间和所需工人数量
        remain_time = graph.nodes[task_node]['remain_time'].seconds / 3600
        required_workers = graph.nodes[task_node]['remain_allocate']

        # 计算分配给该任务的工人数量
        assigned_workers = individual.get(task_id, [])
        # print(assigned_workers)
        num_assigned_workers = len(assigned_workers)

        for worker_id in assigned_workers:
            if is_worker_assigned(worker_id, task_id, individual):
                reward, quality = get_edge_weight(worker_id, task_id, graph)
                total_reward += reward
                total_quality += quality

        if remain_time < 1 and num_assigned_workers < required_workers:
            total_reward += 1000
            total_quality -= 5

    # print([total_reward, -total_quality])

    # 返回适应度向量 [总质量, -总报酬]（负号用于最小化）
    return [total_quality, -total_reward]


# 辅助函数定义
def is_worker_assigned(worker_id, task_id, individual):
    return task_id in individual and worker_id in individual[task_id]


def get_edge_weight(worker_id, task_id, graph):
    edge = graph.get_edge_data(f"worker_{worker_id}", f"task_{task_id}")
    if edge is not None:
        return edge['weight_reward'], edge['weight_quality']
    else:
        return 0, 0


# 判断个体有效性：有效返回true；无效返回false
# 1.工人只能承担一个任务，任务可以分配的工人有数量限制；2.完成时间有约束；3.质量约束
def is_valid_individual(individual, graph):
    used_workers = [worker_id for worker_ids in individual.values() for worker_id in worker_ids]
    # 检查工人ID是否重复
    if len(used_workers) != len(set(used_workers)):
        return False  # 工人ID重复，个体无效

    # 遍历个体中的每个任务，查看任务分配的工人数量
    for task_id, worker_ids in individual.items():
        # 跳过工人ID为空的任务
        if not worker_ids:
            continue

        task_node = f"task_{task_id}"
        # 获取任务结点所有属性
        attributes = graph.nodes[task_node]['attributes']

        if attributes.task_type in ['TypeA', 'TypeB']:
            if len(worker_ids) > 1:
                return False
        else:
            if len(worker_ids) > attributes.num_area - len(attributes.assigned_workers):
                return False

        for worker_id in worker_ids:
            worker_node = f"worker_{worker_id}"

            # 获取边的权重
            edge = graph.get_edge_data(worker_node, task_node)
            if edge is None:
                return False  # 不存在这条边，分配无效

    return True


def repair_invalid(individual, graph):
    # 用于存储每个工人的任务（任务ID）及其完成时间和边权重
    worker_task_mapping = {}

    # 创建一个新的字典以存储有效的任务
    valid_individual = {}

    # 记录哪些工人已经被分配了任务
    assigned_workers = set()

    for task_id, worker_ids in individual.items():
        if not worker_ids:
            continue

        task_node = f"task_{task_id}"

        # 存储有效的工人及其相关信息
        worker_times = {}
        worker_weights = {}

        for worker_id in worker_ids:
            worker_node = f"worker_{worker_id}"
            # 获取边的权重 `weight_total_time`
            edge = graph.get_edge_data(worker_node, task_node)

            if edge is None:
                individual[task_id].remove(worker_id)  # 无边连接，删去该工人id
                continue

            # weight_reward = edge['weight_reward']
            weight_quality = edge['weight_quality']

            if worker_id not in worker_task_mapping:
                worker_task_mapping[worker_id] = (task_id, weight_quality)
            else:
                # 比较工人已分配的任务的完成时间
                existing_task_id, existing_weight = worker_task_mapping[worker_id]
                if weight_quality > existing_weight:
                    worker_task_mapping[worker_id] = (task_id, weight_quality)

    # print(f"保存映射为：{worker_task_mapping}")
    # 遍历 worker_task_mapping
    for worker_id, (task_id, weight_reward) in worker_task_mapping.items():
        task_node = f"task_{task_id}"
        # 获取任务节点的最大工人数量要求
        max_required_workers = graph.nodes[task_node]['remain_allocate']

        # 如果任务 ID 不在个体中，则初始化
        if task_id not in valid_individual:
            valid_individual[task_id] = []

        # 检查当前任务中工人的数量
        if len(valid_individual[task_id]) < max_required_workers and worker_id not in assigned_workers:
            # 添加工人到任务
            valid_individual[task_id].append(worker_id)
            assigned_workers.add(worker_id)
    # print(f'重新分配后的结果为：{valid_individual}')

    # 再分配操作
    final_individual = reallocate_individual(valid_individual, graph)

    return final_individual


def reallocate_individual(valid_individual, graph):
    # 获取所有工人节点和任务节点
    # worker_nodes = [node for node in graph.nodes if node.startswith("worker_")]
    task_nodes = [node for node in graph.nodes if node.startswith("task_")]

    # 用于存储最终的分配结果
    final_individual = valid_individual.copy()  # 基于已有个体进行拷贝

    # 收集现有个体中所有分配的工人ID
    existing_worker_ids = {worker_id for worker_ids in valid_individual.values() for worker_id in worker_ids}

    # 遍历每个任务ID
    for task_node in task_nodes:
        # 获取任务id
        task_id = int(task_node.split("_")[1])

        # 检查任务节点是否在图中
        if task_node not in graph.nodes:
            continue  # 如果任务节点不存在，跳过

        # 获取最大可分配工人数
        max_required_workers = graph.nodes[task_node]['remain_allocate']
        remain_time = graph.nodes[task_node]['remain_time']

        # 获取当前任务的已分配工人
        current_assigned_workers = final_individual.get(task_id, [])

        # 如果当前任务的工人数量已达到上限，跳过
        if len(current_assigned_workers) >= max_required_workers:
            continue

        # 获取与任务节点相连的工人节点及其权重
        worker_edges = graph.edges(task_node)

        # 生成工人列表，包含工人ID和边权重
        worker_weight_pairs = []
        for edge in worker_edges:
            worker_node = edge[0] if edge[1] == task_node else edge[1]
            worker_id = int(worker_node.split('_')[1])  # 提取工人ID

            # 获取边的权重
            edge_data = graph.get_edge_data(worker_node, task_node)
            # weight_reward = edge_data['weight_reward']
            weight_quality = edge_data['weight_quality']
            weight_travel_time = edge_data['weight_travel_time']

            # 添加到工人列表中
            worker_weight_pairs.append((worker_id, weight_quality, weight_travel_time))

        # 按照边权重进行排序（权重越大越好）
        worker_weight_pairs.sort(key=lambda x: x[1], reverse=True)

        # 选择合适的工人加入到最终个体中
        for worker_id, _, weight_travel_time in worker_weight_pairs:
            # 确保工人ID不在现有分配中，并且未达到最大工人数
            if worker_id not in existing_worker_ids and len(current_assigned_workers) < max_required_workers:
                current_assigned_workers.append(worker_id)
                # 检查条件：如果剩余时间大于2，只分配travel_time < 1的工人
                if remain_time > timedelta(hours=2) and weight_travel_time >= 1:
                    continue  # 跳过不符合条件的工人

                # 更新现有工人ID集合，以保证唯一性
                existing_worker_ids.add(worker_id)

        # 更新最终个体
        final_individual[task_id] = current_assigned_workers

    return final_individual


# 普通选择操作——精英保留+锦标赛选择/轮盘赌选择
def selection_origin(population, graph):
    # 根据适应度排序种群
    population.sort(key=lambda indv: calculate_unfitness(indv, graph), reverse=False)
    elite_size = len(population) // 5  # 保留适应度最高的五分之一作为精英
    selected = population[:elite_size]  # 保留精英
    # # 使用轮盘赌法选择剩余的个体
    # fitness_scores = [1 / (calculate_fitness(indv, graph) + 1e-5) for indv in
    #                   population]  # 计算适应度值并反转，使适应度越小越好
    # total_fitness = sum(calculate_fitness(indv, graph) for indv in population)
    # while len(selected) < len(population):
    #     rand_num = random.uniform(0, total_fitness)
    #     cumulative_fitness = 0
    #     for i, indv in enumerate(population):
    #         # cumulative_fitness += calculate_fitness(indv, workers, tasks, current_time)
    #         cumulative_fitness += fitness_scores[i]
    #         if cumulative_fitness > rand_num:
    #             selected.append(indv)
    #             break

    tournament_size = 3
    # 进行锦标赛选择，直到选满所需个体
    while len(selected) < len(population):
        tournament = random.sample(population, tournament_size)  # 随机选择一组个体
        tournament_winner = min(tournament, key=lambda indv: calculate_unfitness(indv, graph))
        selected.append(tournament_winner)  # 选择适应度最高的个体
    return selected


# 非支配排序
def non_dominated_sort_original(population, fitness_scores):
    # 创建一个空列表，存储非支配前沿
    fronts: List[List[int]] = []  # 指定fronts为包含列表的列表
    domination_count = [0] * len(population)
    dominated_set = [[] for _ in range(len(population))]

    # 计算支配关系
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                if dominates(fitness_scores[i], fitness_scores[j]):
                    dominated_set[i].append(j)
                elif dominates(fitness_scores[j], fitness_scores[i]):
                    domination_count[i] += 1

        # 如果个体 i 不被任何其他个体支配，加入第一层
        if domination_count[i] == 0:
            if len(fronts) == 0:
                fronts.append([])  # 创建第一层
            fronts[0].append(i)

    # 进行非支配排序，继续找到所有非支配前沿
    current_front_index = 0

    while current_front_index < len(fronts):
        next_front = []
        for index in fronts[current_front_index]:
            for dominated_index in dominated_set[index]:
                domination_count[dominated_index] -= 1
                if domination_count[dominated_index] == 0:
                    next_front.append(dominated_index)

        # 只记录非支配前沿
        if next_front:
            fronts.append(next_front)

        current_front_index += 1

    # return fronts[0]  # 返回第一非支配前沿的索引
    return fronts  # 返回所有非支配前沿的索引


def non_dominated_sort(population, fitness_scores):
    fronts: List[List[int]] = []
    domination_count = [0] * len(population)
    dominated_set = [[] for _ in range(len(population))]

    # 计算支配关系
    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                if dominates(fitness_scores[i], fitness_scores[j]):
                    dominated_set[i].append(j)
                elif dominates(fitness_scores[j], fitness_scores[i]):
                    domination_count[i] += 1

    for i in range(len(population)):
        if domination_count[i] == 0:
            if len(fronts) == 0:
                fronts.append([])
            fronts[0].append(i)

    current_front_index = 0

    while current_front_index < len(fronts):
        next_front = []
        for index in fronts[current_front_index]:
            for dominated_index in dominated_set[index]:
                domination_count[dominated_index] -= 1
                if domination_count[dominated_index] == 0:
                    next_front.append(dominated_index)

        if next_front:
            fronts.append(next_front)

        current_front_index += 1

    # # 调试信息
    # total_count = sum(len(front) for front in fronts)
    # print(f"Total个体数量:{total_count}，前沿是：{len(fronts)}个" )
    # for front in fronts:
    #     print(f"如果遍历fronts中的元素为:{front}")

    return fronts


def dominates(fitness1, fitness2):
    return all(x >= y for x, y in zip(fitness1, fitness2)) and any(x > y for x, y in zip(fitness1, fitness2))


# 使用拥挤度选择
def crowding_selection(population, fitness, num_select):
    fronts = non_dominated_sort(population, fitness)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= num_select:
            selected.extend(front)
        else:
            # 使用拥挤度选择
            crowding_distances = calculate_crowding_distance(front, fitness)
            sorted_front = sorted(front, key=lambda x: crowding_distances[x], reverse=True)
            selected.extend(sorted_front[:num_select - len(selected)])
            break

    return selected


def calculate_crowding_distance(front, fitness):
    # 计算每个个体的拥挤度距离
    distances = {index: 0 for index in front}

    # 对每个目标进行排序和计算
    for objective in range(len(fitness[0])):
        front.sort(key=lambda x: fitness[x][objective])
        distances[front[0]] = float('inf')  # 边界点
        distances[front[-1]] = float('inf')  # 边界点

        for i in range(1, len(front) - 1):
            distances[front[i]] += (fitness[front[i + 1]][objective] - fitness[front[i - 1]][objective])

    return distances


def pareto_tournament_selection(population, tournament_size, fitness_scores):
    # 进行非支配排序
    fronts = non_dominated_sort(population, fitness_scores)

    selected = []

    # 先选择第一前沿和第二前沿个体
    for front in fronts[:3]:  # 选择前两个前沿，改
        selected.extend(population[i] for i in front)  # 通过下标获取个体

    # print(f"使用非支配前沿后，种群中的个体数量为：{len(selected)}")

    # 如果选出的个体数量未达到种群大小，则进行锦标赛选择
    while len(selected) < len(population):
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament = [population[i] for i in tournament_indices]
        # tournament = random.sample(population, min(tournament_size, len(population)))
        # dominated = [
        # indv for indv in tournament if any(dominates(other, indv, fitness_scores) for other in tournament)
        # ]
        dominated = [indv for indv in tournament if any(
            dominates(fitness_scores[population.index(other)], fitness_scores[population.index(indv)])
            for other in tournament if other != indv)]

        if dominated:
            winner = max(dominated, key=lambda indv: fitness_scores[population.index(indv)][0])
        else:
            # 随机选择一个作为赢家，确保有足够的个体
            winner = random.choice(tournament)

        selected.append(winner)

    return selected[:len(population)]


# 选择每一代最优个体
def select_best_individuals(population, fitness_scores):
    # 计算性价比：质量 / 报酬（避免除零错误）
    def get_cost_effectiveness(idx):
        reward = fitness_scores[idx][0]
        quality = fitness_scores[idx][1]
        if reward == 0:  # 防止除以零
            return float('inf')  # 如果报酬为零，返回一个极大值，表示此个体不可选
        return quality / reward

    fronts = non_dominated_sort(population, fitness_scores)
    # print("Fronts:", fronts)  # 添加调试信息

    # 选择第一个前沿的个体
    if fronts and len(fronts) > 0 and isinstance(fronts[0], list):
        # 使用索引直接获取适应度分数
        best_indices = fronts[0]
        # best_fitness_scores = [fitness_scores[i] for i in best_indices]

        # 找到性价比最高的个体
        best_index = max(best_indices, key=get_cost_effectiveness)
        best_fitness = fitness_scores[best_index]
        return population[best_index], best_fitness

    # 如果没有个体，返回适应度函数计算中报酬最低的个体
    if population:
        best_index = max(range(len(population)), key=get_cost_effectiveness)
        best_fitness = fitness_scores[best_index]
        return population[best_index], best_fitness

    return None, None  # 如果种群为空，返回 None


# 均匀交叉，生成1个新的个体
def uniform_crossover(parent1, parent2):
    child = {}

    # 选择父代工人，并形成子代
    for task_id in parent1:
        worker_id1 = parent1.get(task_id, [])
        worker_id2 = parent2.get(task_id, [])

        # 根据50%的概率选择父代1或父代2的工人
        if worker_id2 is None or (worker_id1 and random.random() > 0.5):
            child[task_id] = worker_id1
        else:
            child[task_id] = worker_id2

    return child


def random_uniform_crossover(parent1, parent2):
    child = {}

    for task_id in parent1:
        # 获取父代中的工人 ID
        worker_ids1 = set(parent1.get(task_id, []))
        worker_ids2 = set(parent2.get(task_id, []))

        # 合并并去重工人 ID
        combined_worker_ids = list(worker_ids1.union(worker_ids2))
        random.shuffle(combined_worker_ids)  # 打乱顺序

        # 获取父代中较小的工人数量
        limit = min(len(worker_ids1), len(worker_ids2))

        # 将不超过限制的工人 ID 加入子代
        child[task_id] = combined_worker_ids[:limit]

    return child


def mutation(individual, mutation_rate, graph):
    # 根据变异率决定是否变异
    if random.random() > mutation_rate:
        return individual

    used_workers = [worker_id for worker_ids in individual.values() for worker_id in worker_ids]

    no_empty_tasks = [task_id for task_id, worker_ids in individual.items() if worker_ids]
    if not len(no_empty_tasks):
        return individual

    # 随机选择一个任务id
    tid = random.choice(no_empty_tasks)
    worker_ids = individual[tid]

    if not worker_ids:
        return individual  # 确保任务至少有一个工人

    # 随机选择一个工人id
    wid = random.choice(worker_ids)
    worker_ids.remove(wid)

    # 寻找符合条件的工人
    task_node = f"task_{tid}"

    available_workers = set(graph.neighbors(task_node)) - set(used_workers)

    if available_workers:
        new_worker_id = random.choice(list(available_workers))
        new_worker_id = int(new_worker_id.split('_')[1])
        individual[tid].append(new_worker_id)

    return individual


# 遗传算法流程
def pareto_quality_my_ga(workers, tasks, current_time, time_slot,
                         population_size, num_generations, mutation_rate, tournament_size):
    graph = build_graph(workers, tasks, current_time, time_slot)
    # visualize_graph(graph)
    # 初始化种群
    population = initialize_population(population_size, graph)
    # print("初始化完成")
    # print(f"种群所有个体：{population}")

    # 用于记录每一代的最佳适应度
    best_fitness_history = []

    # 保存迭代中最优解及其适应度
    best_individual = population[0]
    best_fitness = calculate_unfitness(best_individual, graph)

    for generation in range(num_generations):
        print(f"当前为第{generation + 1}代，个体数：{len(population)}")
        # 计算每个个体的适应度
        fitness_scores = [calculate_unfitness(indiv, graph) for indiv in population]
        # print(f"种群所有适应度为：{fitness_scores}")

        # 记录当前代最优适应度
        current_best_individual, current_best_fitness = select_best_individuals(population, fitness_scores)
        best_fitness_history.append(current_best_fitness)

        # 检查种群中是否存在有效的个体
        if any(fitness_scores):
            # 选择适应度较高的个体
            next_generation = pareto_tournament_selection(population, tournament_size, fitness_scores)
            # print(f"下一代个体数量：{len(next_generation)}")
            # print("选择完成")

            new_population = []

            # 交叉生成新的个体
            for _ in range(len(next_generation)):
                parent1 = random.choice(next_generation)
                parent2 = random.choice(next_generation)
                # print(f"父代为：{parent1}和{parent2}")
                child = uniform_crossover(parent1, parent2)
                reallocate_individual(child, graph)

                if not is_valid_individual(child, graph):
                    repair_invalid(child, graph)
                new_population.append(child)
                # print(f"修复后的子代为：{child}")
            # print("交叉完成")

            # 变异
            new_population = [mutation(individual, mutation_rate, graph) for individual in
                              new_population]
            # print("变异完成")

            repaired_population = []
            # 修复无效个体
            for indiv in new_population:
                if not is_valid_individual(indiv, graph):
                    repaired_indiv = repair_invalid(indiv, graph)
                    repaired_population.append(repaired_indiv)
                else:
                    repaired_population.append(indiv)

            # 更新种群
            repaired_fitness_values = [calculate_unfitness(indiv, graph) for indiv in repaired_population]

            fronts = non_dominated_sort(repaired_population, repaired_fitness_values)
            # print(f"开始更新种群")

            # print(f"当前前沿为：{fronts}")
            new_population_indices = []
            for front in fronts:
                # print(f"front为：{front}")
                if len(new_population_indices) + len(front) <= population_size:
                    new_population_indices.extend(front)
                    # print(f"新的种群中个体下标为：{new_population_indices}")

            # 根据下标构建新的种群
            population = [repaired_population[i] for i in new_population_indices]
            # print(f"更新后的种群大小为：{len(population)}")
            # 更新完成

            # 更新最优解
            current_best_individual, current_best_fitness = select_best_individuals(population, repaired_fitness_values)
            if current_best_fitness[0] > best_fitness[0]:
                best_fitness = current_best_fitness
                best_individual = current_best_individual
            print(f'{generation}最佳适应度：{best_fitness}，该个体为：{best_individual}')
        else:
            break

    return best_individual, best_fitness, best_fitness_history, graph
