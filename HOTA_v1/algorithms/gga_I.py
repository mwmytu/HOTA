"""
GGA-I算法，基于我的遗传算法直接实现的，改了些细节，肯定比原来要快，但是思路没变
"""
import networkx as nx
from utils import *
# from visualize import visualize_graph


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
            # if not worker.is_available(current_time):  # 如果工人当前不可用，跳过此工人
            #     continue

            if task.task_type in ['TypeA', 'TypeB']:
                task_start = to_dt(task.start_time) + timedelta(hours=task.interval * len(task.assigned_workers))
                worker_start = calculate_worker_start_time(worker, current_time)
                start_task = worker_start if worker_start > task_start else task_start
            else:
                start_task = calculate_worker_start_time(worker, current_time)

            # 计算到达并完成任务需要的时间
            travel_time = calculate_travel_time(worker, task)
            # if travel_time > 1:
            #     continue
            sensing_time = calculate_pair_sensing_time(worker, task)
            total_time = travel_time + sensing_time

            # 计算边的权重
            weight_start = start_task
            weight_travel_time = travel_time
            weight_distance = calculate_distance(worker, task)
            weight_sensing_time = sensing_time
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
                                   weight_start=weight_start,
                                   weight_travel_time=weight_travel_time, weight_distance=weight_distance,
                                   weight_sensing_time=weight_sensing_time, weight_reward=weight_reward,
                                   weight_quality=weight_quality)
            else:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= to_dt(task.end_time)):
                    if weight_quality >= task.quality_threshold:
                        g.add_edge("worker_" + str(worker.id), "task_" + str(task.id),
                                   weight_start=weight_start,
                                   weight_travel_time=weight_travel_time, weight_distance=weight_distance,
                                   weight_sensing_time=weight_sensing_time, weight_reward=weight_reward,
                                   weight_quality=weight_quality)

    return g


# 距离最近
def nearest_first(graph):
    # 获取所有工人节点和任务节点
    # worker_nodes = [node for node in graph.nodes if node.startswith("worker_")]
    task_nodes = [node for node in graph.nodes if node.startswith("task_")]
    # print(task_nodes)

    # 结果字典，任务ID映射到工人ID列表
    task_assignments = {int(task_node.split("_")[1]): [] for task_node in task_nodes}
    # print(task_assignments)

    # 用于跟踪已分配的工人ID
    assigned_worker_ids = set()

    # 收集所有工人与任务之间的边及其权重
    edges = []
    for task_node in task_nodes:
        task_id = int(task_node.split("_")[1])
        remain_allocate = graph.nodes[task_node].get('remain_allocate', 0)

        # 获取邻接工人节点
        worker_neighbors = [
            worker_node for worker_node in graph.neighbors(task_node) if worker_node.startswith("worker_")
        ]

        # 如果没有邻接工人，跳过此任务
        if not worker_neighbors:
            continue

        for worker_node in worker_neighbors:
            worker_id = int(worker_node.split("_")[1])
            # 获取边的权重
            edge_data = graph[worker_node][task_node]
            distance = edge_data.get('weight_distance', float('inf'))  # 默认值防止异常

            # 添加到边列表
            edges.append((task_id, worker_id, distance, remain_allocate))

    # 按边的权重升序排序
    edges.sort(key=lambda x: x[2])
    # print(edges)

    # 开始分配工人
    for task_id, worker_id, weight, remain_allocate in edges:
        # 检查工人是否已被分配
        if worker_id in assigned_worker_ids:
            continue

        # 检查任务的可分配数量
        if len(task_assignments[task_id]) < remain_allocate:
            # print(f"{task_id}个体中数量：{len(task_assignments[task_id])}，需要的数量为：{remain_allocate}")
            task_assignments[task_id].append(worker_id)
            # print(task_assignments)
            assigned_worker_ids.add(worker_id)  # 记录已分配的工人

    return task_assignments


# 原来是随机，现在改距离最近优先分配了
# 不行，贪心之后重复度太高
def create_individual_from_graph(graph):
    # 获取所有工人节点和任务节点
    # worker_nodes = [node for node in graph.nodes if node.startswith("worker_")]
    task_nodes = [node for node in graph.nodes if node.startswith("task_")]
    # print(task_nodes)

    # 初始化个体
    individual = {int(task_node.split("_")[1]): [] for task_node in task_nodes}
    # print(individual)

    # 记录已分配的工人
    assigned_workers = set()

    # 打乱任务后进行分配
    random.shuffle(task_nodes)
    for task_node in task_nodes:
        task_id = int(task_node.split("_")[1])
        # if task_id not in individual:
        #     individual[task_id] = []

        # 计算该任务所需工人数
        required_workers_count = graph.nodes[task_node]['remain_allocate']

        # 找到所有与任务节点相连的工人节点
        possible_workers = [
            (n, graph[n][task_node]['weight_distance']) for n in graph.neighbors(task_node) if n.startswith("worker_")
        ]
        # print(possible_workers)

        if possible_workers:
            # 对工人节点进行随机排序，以确保分配的公平性
            random.shuffle(possible_workers)

            # 按照距离权重进行排序
            # possible_workers.sort(key=lambda x: x[1])

            for worker_node, weight_distance in possible_workers:
                worker_id = int(worker_node.split("_")[1])  # 转换为整数
                if worker_id not in assigned_workers and len(individual[task_id]) < required_workers_count:
                    individual[task_id].append(worker_id)
                    assigned_workers.add(worker_id)

                # 如果已满足所需工人数，则跳出循环
                if len(individual[task_id]) >= required_workers_count:
                    break
        # else:
        #     individual[task_id] = []

    return individual


def initialize_population(population_size, graph):
    population = []
    # 加入一个nearest-first解
    assignments = nearest_first(graph)
    print(f"nearest-first解为：{assignments}")
    population.append(assignments)
    for _ in range(population_size - 1):
        individual = create_individual_from_graph(graph)
        # print(individual)
        population.append(individual)
    return population


# 适应度纯靠距离+惩罚项
def calculate_fitness(individual, graph):
    fitness = 0

    # 获取所有任务
    tasks = [int(node[5:]) for node in graph.nodes if node.startswith('task')]

    for task_id in tasks:
        task_node = f"task_{task_id}"
        # 获取任务的剩余时间和所需工人数量
        remain_time = graph.nodes[task_node]['remain_time'].seconds / 3600
        required_workers = graph.nodes[task_node]['remain_allocate']

        # 计算分配给该任务的工人数量
        assigned_workers = individual.get(task_id, [])
        num_assigned_workers = len(assigned_workers)

        for worker_id in assigned_workers:
            if is_worker_assigned(worker_id, task_id, individual):
                distance = get_distance(worker_id, task_id, graph)
                fitness += distance  # 适应度靠距离计算

        if remain_time < 1 and num_assigned_workers < required_workers:
            fitness += 1000

    return fitness


# 辅助函数定义
def is_worker_assigned(worker_id, task_id, individual):
    return task_id in individual and worker_id in individual[task_id]


def get_distance(worker_id, task_id, graph):
    edge = graph.get_edge_data(f"worker_{worker_id}", f"task_{task_id}")
    return edge['weight_distance'] if edge else 1000


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


# 1.工人只能承担一个任务（工人id只能出现一次，随时检查）
# 2.任务可以分配的工人有数量限制（工人id不能重复）
# 3.完成时间有约束、质量约束（不存在可分配关系直接删除）
# 4.再分配
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
        # worker_times = {}
        worker_weights = {}

        for worker_id in worker_ids:
            worker_node = f"worker_{worker_id}"
            # 获取边的权重
            edge = graph.get_edge_data(worker_node, task_node)

            if edge is None:
                individual[task_id].remove(worker_id)  # 无边连接，删去该工人id
                continue
            else:
                weight_distance = edge['weight_distance']
                worker_weights[worker_id] = weight_distance
                if worker_id not in worker_task_mapping:
                    worker_task_mapping[worker_id] = (task_id, weight_distance)
                else:
                    # 比较工人已分配的任务的完成时间
                    existing_task_id, existing_weight = worker_task_mapping[worker_id]
                    if weight_distance < existing_weight:
                        worker_task_mapping[worker_id] = (task_id, weight_distance)

    # print(f"保存映射为：{worker_task_mapping}")
    # 遍历 worker_task_mapping
    for worker_id, (task_id, weight_distance) in worker_task_mapping.items():
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
            weight_distance = edge_data['weight_distance']

            # 添加到工人列表中
            worker_weight_pairs.append((worker_id, weight_distance))

        # 按照边权重——距离进行排序（权重越小越好）
        worker_weight_pairs.sort(key=lambda x: x[1])

        # 选择合适的工人加入到最终个体中
        for worker_id, _ in worker_weight_pairs:
            # 确保工人ID不在现有分配中，并且未达到最大工人数
            if worker_id not in existing_worker_ids and len(current_assigned_workers) < max_required_workers:
                current_assigned_workers.append(worker_id)

                # 更新现有工人ID集合，以保证唯一性
                existing_worker_ids.add(worker_id)

        # 更新最终个体
        final_individual[task_id] = current_assigned_workers

    return final_individual


# 选择操作：直接用的锦标赛选择
def selection(population, graph):
    # 计算所有个体的适应度
    fitness_values = [calculate_fitness(ind, graph) for ind in population]

    tournament_size = 3
    selected_individuals = []

    for _ in range(len(population)):
        # 从种群中随机选择一组个体进行锦标赛
        tournament_individuals = random.sample(population, tournament_size)
        tournament_fitness_values = [fitness_values[population.index(ind)] for ind in tournament_individuals]

        # 找到适应度最好的个体
        best_individual = tournament_individuals[tournament_fitness_values.index(min(tournament_fitness_values))]
        selected_individuals.append(best_individual)

    return selected_individuals


# 单点交叉
def crossover(parent1, parent2):
    # 随机选择交叉点
    crossover_point = random.choice(list(parent1.keys()))
    child1 = {}
    child2 = {}

    for task_id in parent1:
        if task_id < crossover_point:
            child1[task_id] = parent1[task_id]
            child2[task_id] = parent2[task_id]
        else:
            child1[task_id] = parent2[task_id]
            child2[task_id] = parent1[task_id]

    return child1, child2


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


def greedy_genetic_algorithm(workers, tasks, current_time, population_size=50, num_generations=50, mutation_rate=0.1):
    graph = build_bipartite_graph(workers, tasks, current_time)
    # 初始化种群
    population = initialize_population(population_size, graph)
    # print(population)

    # 用于记录每一代的最佳适应度
    best_fitness_history = []

    # 保存迭代中最优解及其适应度
    best_individual = population[0]
    best_fitness = float('inf')

    for generation in range(num_generations):
        # print(f"当前第{generation+1}代")
        # 计算每个个体的适应度
        fitness_values = [calculate_fitness(ind, graph) for ind in population]
        # print(fitness_values)

        # 记录当前代最优适应度
        current_best_fitness = min(fitness_values)
        best_fitness_history.append(current_best_fitness)

        # 选择适应度较高的个体
        next_generation = selection(population, graph)

        # 交叉生成新的个体
        new_population = []
        for i in range(len(next_generation) // 2):
            parent1 = random.choice(next_generation)
            parent2 = random.choice(next_generation)
            child1, child2 = crossover(parent1, parent2)
            if not is_valid_individual(child1, graph):
                repair_invalid(child1, graph)
            new_population.append(child1)

            reallocate_individual(child2, graph)
            if not is_valid_individual(child2, graph):
                repair_invalid(child2, graph)
            new_population.append(child2)

        # 变异
        new_population = [mutation(individual, mutation_rate, graph) for individual in
                          new_population]

        repaired_population = []
        # 修复无效个体
        for ind in new_population:
            if not is_valid_individual(ind, graph):
                repaired_ind = repair_invalid(ind, graph)
                repaired_population.append(repaired_ind)
            else:
                repaired_population.append(ind)

        population = repaired_population
        population.sort(key=lambda ind: calculate_fitness(ind, graph))
        population = population[:population_size]

        if population:
            current_best_individual = population[0]
            # 更新最优个体
            if calculate_fitness(current_best_individual, graph) < best_fitness:
                best_individual = current_best_individual
                best_fitness = calculate_fitness(best_individual, graph)

    return best_individual, best_fitness, best_fitness_history
