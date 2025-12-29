"""
考虑最小化报酬以及最大化质量，实现pareto最优
"""
from typing import List

import networkx as nx
from matplotlib import pyplot as plt

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

    # 将任务分为两组：剩余时间小于1的任务和其他任务
    tasks_with_remain_time_less_than_1 = []
    tasks_with_remain_time_1_or_more = []

    for task_id in task_ids:
        task_node = f"task_{task_id}"
        remain_time = graph.nodes[task_node]['remain_time']

        if remain_time < timedelta(hours=1):
            tasks_with_remain_time_less_than_1.append(task_id)
        else:
            tasks_with_remain_time_1_or_more.append(task_id)

    # 如果有剩余时间小于 1 的任务，优先分配工人
    if tasks_with_remain_time_less_than_1:
        for task_id in tasks_with_remain_time_less_than_1:
            task_node = f"task_{task_id}"
            required_workers_count = graph.nodes[task_node]['remain_allocate']
            possible_workers = []

            # 获取该任务所有可用工人
            for n in graph.neighbors(task_node):
                if n.startswith("worker_"):
                    worker_id = int(n.split("_")[1])
                    weight_reward = graph[task_node][n]['weight_reward']
                    weight_quality = graph[task_node][n]['weight_quality']
                    weight_travel_time = graph[task_node][n]['weight_travel_time']

                    # 假设你有一个工人的剩余时间
                    worker_remain_time = graph.nodes[n]['remain_time']

                    # 如果该工人的 remain_time 满足条件，则加入可选工人
                    # if worker_remain_time >= timedelta(hours=2) and weight_travel_time > 1:
                    #     continue

                    # 将工人加入候选列表
                    possible_workers.append((n, weight_reward, weight_quality))

            # 对可用工人进行排序，可以按需要的标准排序
            possible_workers.sort(key=lambda x: (x[1], x[2]), reverse=True)  # 根据权重排序

            # 为任务分配工人
            for worker, weight_reward, weight_quality in possible_workers:
                worker_id = int(worker.split("_")[1])  # 转换为整数
                if worker_id not in assigned_workers and len(individual[task_id]) < required_workers_count:
                    individual[task_id].append(worker_id)
                    assigned_workers.add(worker_id)

                # 如果已满足工人数要求，跳出循环
                if len(individual[task_id]) >= required_workers_count:
                    break
            else:
                # 如果无法为任务分配足够工人，清空任务的分配
                individual[task_id] = []

    # 接下来处理剩余时间大于等于 1 的任务，随机打乱顺序后分配工人
    if tasks_with_remain_time_1_or_more:
        random.shuffle(tasks_with_remain_time_1_or_more)  # 随机打乱任务顺序

        for task_id in tasks_with_remain_time_1_or_more:
            task_node = f"task_{task_id}"
            required_workers_count = graph.nodes[task_node]['remain_allocate']
            possible_workers = []

            # 获取该任务所有可用工人
            for n in graph.neighbors(task_node):
                if n.startswith("worker_"):
                    worker_id = int(n.split("_")[1])
                    weight_reward = graph[task_node][n]['weight_reward']
                    weight_quality = graph[task_node][n]['weight_quality']
                    weight_travel_time = graph[task_node][n]['weight_travel_time']

                    # 假设你有一个工人的剩余时间
                    worker_remain_time = graph.nodes[n]['remain_time']

                    # 如果该工人的 remain_time 满足条件，则加入可选工人
                    if worker_remain_time >= timedelta(hours=2) and weight_travel_time > 1:
                        continue

                    # 将工人加入候选列表
                    possible_workers.append((n, weight_reward, weight_quality))

            # 对可用工人进行排序，可以按需要的标准排序
            possible_workers.sort(key=lambda x: (x[1], x[2]), reverse=True)  # 根据权重排序

            # 为任务分配工人
            for worker, weight_reward, weight_quality in possible_workers:
                worker_id = int(worker.split("_")[1])  # 转换为整数
                if worker_id not in assigned_workers and len(individual[task_id]) < required_workers_count:
                    individual[task_id].append(worker_id)
                    assigned_workers.add(worker_id)

                # 如果已满足工人数要求，跳出循环
                if len(individual[task_id]) >= required_workers_count:
                    break
            else:
                # 如果无法为任务分配足够工人，清空任务的分配
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

        # if remain_time < 1 and num_assigned_workers < required_workers:
        #     total_reward += 1000
        #     total_quality -= 5

    # print([total_reward, -total_quality])

    # 返回适应度向量 [总报酬, -总质量]（负号用于最大化）
    return [total_reward, -total_quality]


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

            weight_reward = edge['weight_reward']
            weight_quality = edge['weight_quality']
            # weight = weight_quality / weight_reward
            weight = weight_quality * weight_quality / weight_reward     # 质量^2/报酬，越高越好，优先考虑质量

            if worker_id not in worker_task_mapping:
                worker_task_mapping[worker_id] = (task_id, weight)
            else:
                # 比较工人已分配的任务的完成时间,应该是比值小的更好
                existing_task_id, existing_weight = worker_task_mapping[worker_id]
                if weight > existing_weight:
                    worker_task_mapping[worker_id] = (task_id, weight)

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

    return valid_individual


# 将任务根据难度和时间进行排序后，再分配

def plot_pareto_front(population, fronts, graph):
    # 提取第一前沿的个体
    first_front_indices = fronts[0]  # 获取第一前沿的个体下标

    # 根据索引提取个体的适应度
    rewards = [calculate_unfitness(population[i], graph)[0] for i in first_front_indices]
    qualities = [-calculate_unfitness(population[i], graph)[1] for i in first_front_indices]  # 恢复质量为正数

    # 绘制第一前沿的散点图
    plt.scatter(rewards, qualities, color='b', label='Pareto Front')
    plt.xlabel('Reward')
    plt.ylabel('Quality')
    plt.title('Pareto Front - Reward vs Quality')
    plt.legend()
    plt.show()


def dominates_original(indv1, indv2, graph):
    # 计算个体的适应度得分
    fitness1 = calculate_unfitness(indv1, graph)
    fitness2 = calculate_unfitness(indv2, graph)
    return all(x <= y for x, y in zip(fitness1, fitness2)) and any(x < y for x, y in zip(fitness1, fitness2))


def dominates(fitness1, fitness2):
    return all(x <= y for x, y in zip(fitness1, fitness2)) and any(x < y for x, y in zip(fitness1, fitness2))


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
            winner = min(dominated, key=lambda indv: fitness_scores[population.index(indv)][0])
        else:
            # 随机选择一个作为赢家，确保有足够的个体
            winner = random.choice(tournament)

        selected.append(winner)

    return selected[:len(population)]


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

    # 初始化第一前沿
    # for i in range(len(population)):
    #     if domination_count[i] == 0:
    #         fronts[0].append(i)

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


# 选择每一代最优个体，最小报酬的那个
def find_best_individual(population, fitness_scores):
    # 非支配排序后，获取第一前沿（fronts[0]）
    fronts = non_dominated_sort(population, fitness_scores)
    first_front = fronts[0]

    # 定义一个函数来计算质量/报酬比值
    def quality_to_reward_ratio(fitness):
        reward = fitness[0]  # 报酬
        quality = -fitness[1]  # 质量，因质量是负值，所以取负值
        return quality / reward if reward != 0 else float('inf')  # 避免除以零

    # 初始化最佳个体和比值
    best_individual_idx = -1  # 初始值设置为 -1，表示还没有找到最优个体
    best_ratio = -float('inf')  # 设置初始最小比值

    for individual_idx in first_front:
        ratio = quality_to_reward_ratio(fitness_scores[individual_idx])
        if ratio > best_ratio:
            best_ratio = ratio
            best_individual_idx = individual_idx

    return population[best_individual_idx], fitness_scores[best_individual_idx]


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


def uniform_crossover_with_probability(parent1, parent2, crossover_rate):
    # crossover_prob = 0.4
    child = {}

    for task_id in parent1:
        # 生成一个随机数，决定是否进行交叉
        if random.random() <= crossover_rate:
            worker_id1 = parent1.get(task_id, [])
            worker_id2 = parent2.get(task_id, [])

            # 根据50%的概率选择父代1或父代2的工人
            if worker_id2 is None or (worker_id1 and random.random() > 0.5):
                child[task_id] = worker_id1
            else:
                child[task_id] = worker_id2
        else:
            # 如果不交叉，直接将父代复制给子代
            child[task_id] = parent1.get(task_id, [])

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
def pareto_my_ga_no_repair(workers, tasks, current_time, time_slot,
                           population_size, num_generations, crossover_rate, mutation_rate, tournament_size):
    graph = build_graph(workers, tasks, current_time, time_slot)
    # visualize_graph(graph)
    # 初始化种群
    population = initialize_population(population_size, graph)
    # print("初始化完成")
    # print(f"BPGA-NR种群所有个体：{population}")

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
                child = uniform_crossover_with_probability(parent1, parent2, crossover_rate)

                new_population.append(child)

                # if is_valid_individual(child, graph):
                #     new_population.append(child)
                # print(f"修复后的子代为：{child}")
            # print("交叉完成")
            # print(f"经过交叉后种群为：{new_population}")

            # 变异
            new_population = [mutation(individual, mutation_rate, graph) for individual in
                              new_population]
            # print("变异完成")

            # 保留有效个体
            new_population = [indiv for indiv in new_population
                              if is_valid_individual(indiv, graph)]
            print(f'有效个体数量：{len(new_population)}')

            # 更新种群
            population = new_population  # 有选出的个体
            while len(population) < population_size:
                new_individual = create_individual_from_graph(graph)
                population.append(new_individual)

            print(f"最后的种群大小:{len(population)}")

            new_fitness_value = [calculate_unfitness(indiv, graph) for indiv in population]
            # fronts = non_dominated_sort(population, new_fitness_value)

            # 更新最优解
            current_best_individual, current_best_fitness = find_best_individual(population, new_fitness_value)
            best_fitness_history.append(current_best_fitness)

            # 为避免零分母，首先检查分母是否为零
            def safe_divide(numerator, denominator):
                if denominator == 0:
                    return float('inf')  # 或者返回一个非常大的值，表示此解不可取
                return numerator / denominator

            current_best_fitness_value = safe_divide(current_best_fitness[0], current_best_fitness[1])
            best_fitness_value = safe_divide(best_fitness[0], best_fitness[1])

            # 比较性价比
            if current_best_fitness_value > best_fitness_value:
                best_fitness = current_best_fitness
                best_individual = current_best_individual
            print(f'{generation}最佳适应度：{current_best_fitness}，该个体为：{current_best_individual}')
        else:
            break

    return best_individual, best_fitness, best_fitness_history, graph
