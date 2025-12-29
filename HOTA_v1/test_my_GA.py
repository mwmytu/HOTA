import json
from matplotlib import pyplot as plt
from algorithms.myGA.pareto_my_ga import pareto_my_ga
from algorithms.myGA.pareto_my_ga_no_repair import pareto_my_ga_no_repair
from algorithms.myGA.pareto_my_ga_no_front import pareto_my_ga_no_front
from algorithms.greedy import greedy_bid_assign
from data.all_workers import Worker
from data.all_tasks import Task
from utils import *
from zhibiao import *
from datetime import time, timedelta


def get_data(num_workers, num_tasks, task_mode):
    def load_from_file(filename):
        with open(filename, 'r') as f:
            return json.load(f)

    def select_by_num(data, num_items):
        # 选择前 num_items 个元素
        selected = data[:num_items]
        selected_sorted = sorted(selected, key=lambda x: x["start_time"])  # 按 start_time 排序
        for new_id, item in enumerate(selected_sorted, start=1):  # 重新赋值 id
            item['id'] = new_id
        return selected_sorted

    # 从文件加载数据
    workers_data = load_from_file(f'data/random_bid/workers_data.json')
    tasks_data = load_from_file(f'data/random_bid/{task_mode}_tasks_data.json')

    # 随机选择并重新赋值 id
    selected_workers = select_by_num(workers_data, num_workers)
    selected_tasks = select_by_num(tasks_data, num_tasks)

    # 将字典列表转换回对象列表
    workers = [Worker.from_dict(data) for data in selected_workers]
    tasks = [Task.from_dict(data) for data in selected_tasks]

    return workers, tasks


def update_status_with_graph(individual, workers, tasks, current_time, graph):
    for task_id, worker_ids in individual.items():
        task = get_task_by_id(tasks, task_id)
        if worker_ids:
            for worker_id in worker_ids:
                worker = get_worker_by_id(workers, worker_id)
                task_location = calculate_nearest_location(worker, task)
                if task.task_type in ['TypeA', 'TypeB']:
                    task_start = to_dt(task.start_time) + timedelta(hours=task.interval * len(task.assigned_workers))
                    worker_start = calculate_worker_start_time(worker, current_time)
                    start_time = worker_start if worker_start > task_start else task_start
                else:
                    start_time = calculate_worker_start_time(worker, current_time)

                # worker_node = "worker_" + str(worker_id)
                edge = graph.get_edge_data(f"worker_{worker_id}", f"task_{task_id}")

                reward = edge['weight_reward']
                quality = edge['weight_quality']
                sensing_time = edge['weight_sensing_time']
                travel_time = edge['weight_travel_time']
                expected_finish_time = current_time + timedelta(hours=travel_time + sensing_time)
                # 记录任务信息到工人的assigned_tasks属性中
                task_info = {
                    'task_id': task_id,
                    'task_location': task_location,
                    'start_task': start_time,
                    'travel_time': travel_time,
                    'sensing_time': sensing_time,
                    'expected_finish_time': expected_finish_time,
                    'reward': reward,
                    'quality': quality,
                }
                worker.assigned_tasks.append(task_info)

                # 更新任务
                worker_info = {
                    'worker_id': worker_id,
                    'start_task': start_time,
                    'travel_time': travel_time,
                    'sensing_time': sensing_time,
                    'expected_finish_time': expected_finish_time,
                    'reward': reward,
                    'quality': quality
                }
                task.assigned_workers.append(worker_info)
            # print(task.__dict__)

    return 0


# 运行一次pareto-ga
def run_pareto_ga_once(num_generations, population_size,
                       mutation_rate, tournament_size, num_workers,
                       num_tasks, task_mode, time_slot):
    workers, tasks = get_data(num_workers, num_tasks, task_mode)
    start_time = to_dt(time(14, 0, 0))
    current_time = start_time
    available_workers = [worker for worker in workers if worker.is_available(current_time)]
    # 找到当前时间点，未分配且在时间范围内的任务，直接使用字符串比较
    tasks_to_assign = [task for task in tasks if task.is_to_assign(current_time)]

    print(f"当前时间：{current_time}，可用工人数量：{len(available_workers)}，待分配任务数量：{len(tasks_to_assign)}")

    # 这里调用遗传算法实现
    (best_individual, best_fitness,
     fitness_history, pareto_front_history) = pareto_my_ga(available_workers,
                                                           tasks_to_assign,
                                                           current_time, time_slot,
                                                           population_size=population_size,
                                                           num_generations=num_generations,
                                                           mutation_rate=mutation_rate,
                                                           tournament_size=tournament_size)

    print(f"适应度记录为：{fitness_history}")
    for fit in fitness_history:
        print(f"遍历适应度为：{fit}")

    # 将适应度历史（报酬和负质量）提取出来
    fitness_data = [(fit[0], -fit[1]) for fit in fitness_history]  # fit[0]是报酬，fit[1]是质量
    return best_individual, best_fitness, fitness_data, pareto_front_history


def plot_pareto_fronts(pareto_front_history):
    # 遍历每一代的Pareto前沿
    for generation, fronts in pareto_front_history.items():
        # 将适应度拆分为两个列表
        rewards = [fitness[0] for fitness in fronts]
        qualities = [fitness[1] for fitness in fronts]

        # 创建图形
        plt.figure()
        plt.scatter(rewards, qualities, label=f'Generation {generation}')
        plt.title(f'Pareto Front at Generation {generation}')
        plt.xlabel('Reward')
        plt.ylabel('Negative Quality')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'pareto_front_generation_{generation}.png')  # 保存图像
        plt.close()  # 关闭图形以释放内存


# 运行一次不带修复操作的pareto-ga-no-repair
def run_pareto_ga_no_repair_once(num_generations, population_size, crossover_rate,
                                 mutation_rate, tournament_size, num_workers,
                                 num_tasks,
                                 task_mode, time_slot):
    workers, tasks = get_data(num_workers, num_tasks, task_mode)
    start_time = to_dt(time(14, 0, 0))
    current_time = start_time
    available_workers = [worker for worker in workers if worker.is_available(current_time)]
    # 找到当前时间点，未分配且在时间范围内的任务，直接使用字符串比较
    tasks_to_assign = [task for task in tasks if task.is_to_assign(current_time)]

    print(f"当前时间：{current_time}，可用工人数量：{len(available_workers)}，待分配任务数量：{len(tasks_to_assign)}")
    # 这里调用遗传算法实现
    (best_individual, best_fitness, fitness_history,
     pareto_front_history) = pareto_my_ga_no_repair(available_workers, tasks_to_assign,
                                                    current_time, time_slot,
                                                    population_size=population_size,
                                                    num_generations=num_generations,
                                                    crossover_rate=crossover_rate,
                                                    mutation_rate=mutation_rate,
                                                    tournament_size=tournament_size)
    print(f"适应度记录为：{fitness_history}")
    print(f"个体为：{best_individual}")

    # 将适应度历史（报酬和负质量）提取出来
    fitness_data = [(fit[0], -fit[1]) for fit in fitness_history]  # fit[0]是报酬，fit[1]是质量
    return best_individual, best_fitness, fitness_data, pareto_front_history


# 运行一次不带前沿保留的pareto-ga-no-front
def run_pareto_ga_no_front_once(num_generations, population_size, crossover_rate,
                                mutation_rate, tournament_size, num_workers,
                                num_tasks,
                                task_mode, time_slot):
    workers, tasks = get_data(num_workers, num_tasks, task_mode)
    start_time = to_dt(time(14, 0, 0))
    current_time = start_time
    available_workers = [worker for worker in workers if worker.is_available(current_time)]
    # 找到当前时间点，未分配且在时间范围内的任务，直接使用字符串比较
    tasks_to_assign = [task for task in tasks if task.is_to_assign(current_time)]

    print(f"当前时间：{current_time}，可用工人数量：{len(available_workers)}，待分配任务数量：{len(tasks_to_assign)}")
    # 这里调用遗传算法实现
    best_individual, best_fitness, fitness_history, graph = pareto_my_ga_no_front(available_workers, tasks_to_assign,
                                                                                  current_time, time_slot,
                                                                                  population_size=population_size,
                                                                                  num_generations=num_generations,
                                                                                  crossover_rate=crossover_rate,
                                                                                  mutation_rate=mutation_rate,
                                                                                  tournament_size=tournament_size)
    print(f"适应度记录为：{fitness_history}")
    print(f"个体为：{best_individual}")

    # 将适应度历史（报酬和负质量）提取出来
    fitness_data = [(fit[0], -fit[1]) for fit in fitness_history]  # fit[0]是报酬，fit[1]是质量
    return best_individual, best_fitness, fitness_data


# 运行一次的greedy
def run_greedy(num_workers, num_tasks, task_mode, time_slot):
    workers, tasks = get_data(num_workers, num_tasks, task_mode)
    start_time = to_dt(time(12, 0, 0))
    current_time = start_time
    available_workers = [worker for worker in workers if worker.is_available(current_time)]
    # 找到当前时间点，未分配且在时间范围内的任务，直接使用字符串比较
    tasks_to_assign = [task for task in tasks if task.is_to_assign(current_time)]
    print(f"当前时间：{current_time}，可用工人数量：{len(available_workers)}，待分配任务数量：{len(tasks_to_assign)}")
    assignments = greedy_bid_assign(workers, tasks, current_time, time_slot)

    return assignments


# 参数敏感性
def main_sensitivity_split():
    num_workers = 800
    num_tasks = 100
    time_slot = 30
    task_mode = 'uniform'  # uniform, concentrated, mixed
    num_generations = 100
    tournament_size = 3

    # 设置参数组合
    param_combinations = [
        (50, 0.4, 0.05),
        (50, 0.6, 0.05),
        (50, 0.4, 0.1),
        (50, 0.6, 0.1),
        (100, 0.4, 0.05),
        (100, 0.6, 0.05),
        (100, 0.4, 0.1),
        (100, 0.6, 0.1)
    ]

    # 设定对比色：蓝色、红色、绿色、黄色
    contrast_colors = ['b', 'c', 'r', 'm', 'g', 'y', 'orange', 'purple']  # Blue, Red, Green, Yellow

    # 记录每个实验的适应度历史
    results = {}

    # 执行实验 10 次
    num_trials = 10  # 进行 10 次实验

    for idx, (population_size, crossover_rate, mutation_rate) in enumerate(param_combinations):
        all_fitness_history = []  # 用于保存 10 次实验的 fitness 历史

        for _ in range(num_trials):
            best_individual, best_fitness, fitness_history, pareto_front_history = run_pareto_ga_once(
                num_generations, population_size, crossover_rate, mutation_rate, tournament_size,
                num_workers, num_tasks, task_mode, time_slot
            )
            all_fitness_history.append(fitness_history)

        # 提取 total_reward 和 total_quality 部分
        total_rewards = np.array([[fitness[0] for fitness in trial] for trial in all_fitness_history])
        total_qualities = np.array([[fitness[1] for fitness in trial] for trial in all_fitness_history])

        # 计算每代的平均总奖励和负质量
        avg_total_reward = np.mean(total_rewards, axis=0)
        avg_total_quality = np.mean(total_qualities, axis=0)

        # 保存平均结果
        results[(population_size, crossover_rate, mutation_rate)] = (avg_total_reward, avg_total_quality)

    # 绘制总奖励图
    plt.figure(figsize=(12, 8))
    for idx, ((population_size, crossover_rate, mutation_rate), (avg_total_reward, avg_total_quality)) in enumerate(results.items()):
        plt.plot(avg_total_reward, label=f'$Num$: {population_size}, $P_{{c}}$:{crossover_rate}, '
                                         f'$P_{{m}}$: {mutation_rate}',
                 color=contrast_colors[idx % len(contrast_colors)])

    plt.xlabel('Generation', fontsize=18)
    plt.ylabel('Total Reward', fontsize=18)
    plt.legend(fontsize=14, loc='upper left')
    plt.grid()
    plt.savefig('total_reward_plot.svg', format='svg', dpi=1200)
    plt.show()

    # 绘制负质量图
    plt.figure(figsize=(12, 8))
    for idx, ((population_size, crossover_rate, mutation_rate), (avg_total_reward, avg_total_quality)) in enumerate(results.items()):
        plt.plot(avg_total_quality, label=f'$Num$: {population_size}, $P_{{c}}$:{crossover_rate}, '
                                          f'$P_{{m}}$: {mutation_rate}',
                 color=contrast_colors[idx % len(contrast_colors)])

    plt.xlabel('Generation', fontsize=18)
    plt.ylabel('Total Quality', fontsize=18)
    plt.legend(fontsize=14, loc='upper left')
    plt.grid()
    plt.savefig('total_quality_plot.svg', format='svg', dpi=1200)
    plt.show()

    # 绘制比值图： 比值 = 总奖励 / 负质量
    plt.figure(figsize=(12, 8))
    for idx, ((population_size, crossover_rate, mutation_rate), (avg_total_reward, avg_total_quality)) in enumerate(results.items()):
        reward_quality_ratio = np.divide(avg_total_reward, avg_total_quality, where=(avg_total_quality != 0),
                                         out=np.zeros_like(avg_total_reward))
        plt.plot(reward_quality_ratio, label=f'$Num$: {population_size},  $P_{{c}}$:{crossover_rate}, '
                                             f'$P_{{m}}$: {mutation_rate}',
                 color=contrast_colors[idx % len(contrast_colors)])

    plt.xlabel('Generation', fontsize=18)
    plt.ylabel('Reward/Quality Ratio', fontsize=18)
    plt.legend(fontsize=14, loc='upper left')
    plt.grid()
    plt.savefig('reward_quality_ratio_plot.svg', format='svg', dpi=1200)
    plt.show()


def main_fitness_curve():
    num_workers = 800
    num_tasks = 200
    time_slot = 30
    task_mode = 'mixed'  # uniform, concentrated, mixed
    num_generations = 100
    population_size = 50
    crossover_rate = 0.4
    mutation_rate = 0.1
    tournament_size = 3

    # 假设这些数据是从上述两个函数返回的
    best_individual1, best_fitness1, fitness_data1, pareto_front_history1 = run_pareto_ga_once(num_generations,
                                                                                               population_size,
                                                                                               mutation_rate,
                                                                                               crossover_rate,
                                                                                               tournament_size,
                                                                                               num_workers, num_tasks,
                                                                                               task_mode, time_slot)
    best_individual2, best_fitness2, fitness_data2, _ = run_pareto_ga_no_repair_once(
        num_generations, population_size, crossover_rate,
        mutation_rate, tournament_size,
        num_workers, num_tasks, task_mode,
        time_slot)
    best_individual3, best_fitness3, fitness_data3 = run_pareto_ga_no_front_once(num_generations, population_size,
                                                                                 mutation_rate, crossover_rate,
                                                                                 tournament_size, num_workers,
                                                                                 num_tasks, task_mode,
                                                                                 time_slot)

    # 将报酬和负质量分别提取出来
    reward1 = [fit[0] for fit in fitness_data1]
    quality1 = [fit[1] for fit in fitness_data1]

    reward2 = [fit[0] for fit in fitness_data2]
    quality2 = [fit[1] for fit in fitness_data2]

    reward3 = [fit[0] for fit in fitness_data3]  # 获取第三个数据集
    quality3 = [fit[1] for fit in fitness_data3]  # 获取第三个数据集

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 18})  # 例如设置为 18，或者根据需要调整

    # 创建适应度随时间变化的曲线 - 第一幅图（奖励比较）
    plt.figure(figsize=(10, 6))  # 创建新的图形窗口
    plt.plot(range(len(reward1)), reward1, label='BPGA', color='red', linestyle='-', marker='o')
    plt.plot(range(len(reward2)), reward2, label='BPGA-NR', color='blue', linestyle='-', marker='x')
    plt.plot(range(len(reward3)), reward3, label='BPGA-NF', color='green', linestyle='-', marker='^')  # 第三条线

    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.title('Mixed Distribution')
    plt.legend(loc='lower right')
    plt.grid(True)

    # 保存图像为文件
    plt.savefig(f'reward_comparison_{task_mode}.pdf', dpi=1200)  # 保存为PNG文件
    plt.show()  # 显示第一幅图

    # 创建适应度随时间变化的曲线 - 第二幅图（质量比较）
    plt.figure(figsize=(10, 6))  # 创建新的图形窗口
    plt.plot(range(len(quality1)), quality1, label='BPGA', color='red', linestyle='--')
    plt.plot(range(len(quality2)), quality2, label='BPGA-NR', color='blue', linestyle='--')
    plt.plot(range(len(quality3)), quality3, label='BPGA-NF', color='green', linestyle='--')  # 第三条线

    plt.xlabel('Generation')
    plt.ylabel('Quality')
    plt.title('Mixed Distribution')
    plt.legend(loc='lower right')
    plt.grid(True)

    # 保存图像为文件
    plt.savefig(f'quality_comparison_{task_mode}.pdf', dpi=1200)  # 保存为PNG文件
    plt.show()  # 显示第二幅图

    # 创建报酬与质量比值的曲线 - 第三幅图
    plt.figure(figsize=(10, 6))  # 创建新的图形窗口
    reward_quality_ratio1 = [r / q if q != 0 else 0 for r, q in zip(reward1, quality1)]
    reward_quality_ratio2 = [r / q if q != 0 else 0 for r, q in zip(reward2, quality2)]
    reward_quality_ratio3 = [r / q if q != 0 else 0 for r, q in zip(reward3, quality3)]  # 第三条线

    plt.plot(range(len(reward_quality_ratio1)), reward_quality_ratio1, label='BPGA',
             color='red', linestyle='-', marker='s')
    plt.plot(range(len(reward_quality_ratio2)), reward_quality_ratio2, label='BPGA-NR',
             color='blue', linestyle='-', marker='^')
    plt.plot(range(len(reward_quality_ratio3)), reward_quality_ratio3, label='BPGA-NF',
             color='green', linestyle='-', marker='d')  # 第三条线

    plt.xlabel('Generation')
    plt.ylabel('Reward/Quality Ratio')
    plt.title('Mixed Distribution')
    plt.legend(loc='lower right')
    plt.grid(True)

    # 保存图像为文件
    plt.savefig(f'reward_quality_ratio_comparison_{task_mode}.pdf', dpi=1200)  # 保存为PNG文件
    plt.show()  # 显示第三幅图


if __name__ == '__main__':
    # main_sensitivity_split()
    main_fitness_curve()
