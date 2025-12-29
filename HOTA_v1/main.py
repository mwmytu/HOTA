"""
测试算法运行，总结果运行实现
"""
import csv
import json
import os
from algorithms.greedy import greedy_bid_assign, greedy_quality_reward_ratio
from algorithms.binary_match import maximum_flow_allocation
from algorithms.gga_I import greedy_genetic_algorithm
from algorithms.myGA.pareto_my_ga import pareto_my_ga
from algorithms.myGA.pareto_my_ga_no_repair import pareto_my_ga_no_repair
from data.all_workers import Worker
from data.all_tasks import Task
from utils import *
from zhibiao import *
from datetime import datetime as dt, time, timedelta


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


def run_greedy(num_workers, num_tasks, task_mode, time_slot):
    workers, tasks = get_data(num_workers, num_tasks, task_mode)
    start_time = to_dt(time(8, 0, 0))
    end_time = to_dt(time(20, 0, 0))
    time_step = timedelta(minutes=time_slot)

    current_time = start_time
    while current_time < end_time:
        # 找到当前时间点的可用的工人
        available_workers = [worker for worker in workers if worker.is_available(current_time)]
        # 找到当前时间点，未分配且在时间范围内的任务，直接使用字符串比较
        tasks_to_assign = [task for task in tasks if task.is_to_assign(current_time)]

        # 检查工人或任务是否为空
        if not available_workers or not tasks_to_assign:
            current_time += time_step
            continue

        print(current_time)
        print(f"贪心算法，当前的可用工人数{len(available_workers)}，待分配任务数：{len(tasks_to_assign)}")
        # 使用贪心算法进行任务分配
        assignments = greedy_quality_reward_ratio(available_workers, tasks_to_assign, current_time, time_slot)
        # assignments = greedy_bid_assign(available_workers, tasks_to_assign, current_time, time_slot)

        if any(assignments.values()):
            print(assignments)
            assigned_tasks, assigned_workers = 0, 0
            for task, worker_ids in assignments.items():
                if worker_ids:
                    assigned_tasks += 1
                    assigned_workers += len(worker_ids)
            print(f'分配的工人数{assigned_workers}, 分配的任务数{assigned_tasks}')
            update_status(workers, tasks, assignments, current_time)
        else:
            print("没有合适的分配")

        current_time += time_step  # 确保每次循环都更新当前时间

    print("开始打印没完成的任务")
    for task in tasks:
        if task.task_type in ['TypeA', 'TypeB']:
            if len(task.assigned_workers) < task.sensing_rounds:
                print(task.id)
                print(task.__dict__)
        else:
            if len(task.assigned_workers) < task.num_area:
                print(task.id)
                print(task.__dict__)

    # 指标
    success_task_rate = calculate_success_task_rate(tasks)
    total_distance = calculate_total_distance(workers)
    total_travel_time = calculate_total_travel_time(workers)
    worker_utilization = calculate_worker_utilization(workers)
    total_reward = calculate_total_reward(tasks)
    total_quality = calculate_total_quality(tasks)
    reward_quality_ratio = total_reward / total_quality
    reward_distance_ratio = total_reward / total_distance
    distance_quality_ratio = total_distance / total_quality

    return (success_task_rate, total_distance, total_travel_time, worker_utilization,
            total_reward, total_quality, reward_quality_ratio, reward_distance_ratio, distance_quality_ratio)


def run_maximum_flow(num_workers, num_tasks, task_mode, time_slot):
    workers, tasks = get_data(num_workers, num_tasks, task_mode)
    # print(f'最大流分配算法开始：{id(workers)}, {id(tasks)}')
    start_time = to_dt(time(8, 0, 0))
    end_time = to_dt(time(20, 0, 0))
    time_step = timedelta(minutes=time_slot)

    current_time = start_time
    while current_time < end_time:
        # 找到当前时间点的可用的工人
        available_workers = [worker for worker in workers if worker.is_available(current_time)]
        # 找到当前时间点，未分配且在时间范围内的任务，直接使用字符串比较
        tasks_to_assign = [task for task in tasks if task.is_to_assign(current_time)]
        # 检查工人或任务是否为空
        if not available_workers or not tasks_to_assign:
            current_time += time_step
            continue

        print(current_time)
        print(f"最大流分配算法，当前的可用工人数{len(available_workers)}，待分配任务数：{len(tasks_to_assign)}")
        # 使用最大流分配算法进行任务分配
        assignments = maximum_flow_allocation(available_workers, tasks_to_assign, current_time, time_slot)
        empty = True
        for value in assignments.values():
            if value:
                empty = False
                break

        if not empty:
            print(assignments)
            update_status(workers, tasks, assignments, current_time)

        current_time += time_step  # 确保每次循环都更新当前时间
        # print(f'最大流算法结束：{id(workers)}, {id(tasks)}\n')

    print("开始打印没完成的任务")
    for task in tasks:
        if task.task_type in ['TypeA', 'TypeB']:
            if len(task.assigned_workers) < task.sensing_rounds:
                print(task.id)
                print(task.__dict__)
        else:
            if len(task.assigned_workers) < task.num_area:
                print(task.id)
                print(task.__dict__)

    # 指标
    success_task_rate = calculate_success_task_rate(tasks)
    total_distance = calculate_total_distance(workers)
    total_travel_time = calculate_total_travel_time(workers)
    worker_utilization = calculate_worker_utilization(workers)
    total_reward = calculate_total_reward(tasks)
    total_quality = calculate_total_quality(tasks)
    reward_quality_ratio = total_reward / total_quality
    reward_distance_ratio = total_reward / total_distance
    distance_quality_ratio = total_distance / total_quality

    return (success_task_rate, total_distance, total_travel_time, worker_utilization,
            total_reward, total_quality, reward_quality_ratio, reward_distance_ratio, distance_quality_ratio)


def run_greedy_genetic(num_generations, population_size, mutation_rate, num_workers, num_tasks, task_mode, time_slot):
    workers, tasks = get_data(num_workers, num_tasks, task_mode)
    start_time = to_dt(time(8, 0, 0))
    end_time = to_dt(time(20, 0, 0))
    time_slot = timedelta(minutes=time_slot)

    current_time = start_time
    while current_time < end_time:
        # 找到当前时间点的可用的工人
        available_workers = [worker for worker in workers if worker.is_available(current_time)]
        # 找到当前时间点，未分配且在时间范围内的任务，直接使用字符串比较
        tasks_to_assign = [task for task in tasks if task.is_to_assign(current_time)]
        # 检查工人或任务是否为空
        if not available_workers or not tasks_to_assign:
            current_time += time_slot
            continue

        print(current_time)
        print(f"GGA-I算法，当前的可用工人数{len(available_workers)}，待分配任务数：{len(tasks_to_assign)}")
        # 使用贪心算法进行任务分配
        assignments, best_fitness, best_fitness_history = greedy_genetic_algorithm(available_workers, tasks_to_assign,
                                                                                   current_time,
                                                                                   population_size,
                                                                                   num_generations, mutation_rate)

        empty = True
        for value in assignments.values():
            if value:
                empty = False
                break

        if not empty:
            print(f"分配结果为：{assignments}")
            update_status(workers, tasks, assignments, current_time)

        # plt.plot(best_fitness_history, label=f'Time: {current_time}')

        current_time += time_slot  # 确保每次循环都更新当前时间
    # plt.xlabel('Generation')
    # plt.ylabel('Best Fitness')
    # plt.title('GGA-I Convergence')
    # plt.legend()
    # plt.show()

    for task in tasks:
        if task.task_type in ['TypeA', 'TypeB']:
            if len(task.assigned_workers) < task.sensing_rounds:
                print(task.id)
                print(task.__dict__)
        else:
            if len(task.assigned_workers) < task.num_area:
                print(task.id)
                print(task.__dict__)

    # print(f'GGA-I算法结束：{id(workers)}, {id(tasks)}\n')
    # 指标
    success_task_rate = calculate_success_task_rate(tasks)
    total_distance = calculate_total_distance(workers)
    total_travel_time = calculate_total_travel_time(workers)
    worker_utilization = calculate_worker_utilization(workers)
    total_reward = calculate_total_reward(tasks)
    total_quality = calculate_total_quality(tasks)
    reward_quality_ratio = total_reward / total_quality
    reward_distance_ratio = total_reward / total_distance
    distance_quality_ratio = total_distance / total_quality

    return (success_task_rate, total_distance, total_travel_time, worker_utilization,
            total_reward, total_quality, reward_quality_ratio, reward_distance_ratio, distance_quality_ratio)


def run_my_genetic(num_generations, population_size, mutation_rate, tournament_size,
                   num_workers, num_tasks, task_mode, time_slot):
    workers, tasks = get_data(num_workers, num_tasks, task_mode)
    # print(f'遗传算法开始：{id(workers)}, {id(tasks)}')
    start_time = to_dt(time(8, 0, 0))
    end_time = to_dt(time(20, 0, 0))
    time_step = timedelta(minutes=time_slot)

    current_time = start_time
    while current_time < end_time:
        # 找到当前时间点的可用的工人
        available_workers = [worker for worker in workers if worker.is_available(current_time)]
        # 找到当前时间点，未分配且在时间范围内的任务，直接使用字符串比较
        tasks_to_assign = [task for task in tasks if task.is_to_assign(current_time)]
        # 检查工人或任务是否为空
        # if not available_workers or not tasks_to_assign:
        #     current_time += time_slot
        #     continue

        print(current_time)
        print(f"遗传算法BPGA，当前的可用工人数{len(available_workers)}，待分配任务数：{len(tasks_to_assign)}")
        # 使用遗传算法进行任务分配
        best_individual, best_fitness, best_fitness_history, graph = pareto_my_ga(available_workers, tasks_to_assign,
                                                                                  current_time, time_slot,
                                                                                  population_size, num_generations,
                                                                                  mutation_rate, tournament_size)
        if any(best_individual.values()):
            print(best_individual)
            # fitness = calculate_fitness(best_individual, workers, graph)
            assigned_tasks, assigned_workers = 0, 0
            for task, worker_ids in best_individual.items():
                if worker_ids:
                    assigned_tasks += 1
                    assigned_workers += len(worker_ids)
            # print(f'遗传算法算法分配的工人数{assigned_workers}, 分配的任务数{assigned_tasks}，适应度{fitness}')
            update_status(workers, tasks, best_individual, current_time)
        else:
            print("没有合适的分配")

        # update_task_priority(tasks, best_individual, current_time, time_slot)

        current_time += time_step  # 确保每次循环都更新当前时间
    # plt.xlabel('Generation')
    # plt.ylabel('Best Fitness')
    # plt.title('Genetic Algorithm Convergence')
    # plt.legend()
    # plt.show()
    # print(f'遗传算法结束：{id(workers)}, {id(tasks)}\n')

    print("开始打印没完成的任务")
    for task in tasks:
        if task.task_type in ['TypeA', 'TypeB']:
            if len(task.assigned_workers) < task.sensing_rounds:
                print(task.id)
                print(task.__dict__)
        else:
            if len(task.assigned_workers) < task.num_area:
                print(task.id)
                print(task.__dict__)

    # 指标
    success_task_rate = calculate_success_task_rate(tasks)
    total_distance = calculate_total_distance(workers)
    total_travel_time = calculate_total_travel_time(workers)
    worker_utilization = calculate_worker_utilization(workers)
    total_reward = calculate_total_reward(tasks)
    total_quality = calculate_total_quality(tasks)
    reward_quality_ratio = total_reward / total_quality
    reward_distance_ratio = total_reward / total_distance
    distance_quality_ratio = total_distance / total_quality

    return (success_task_rate, total_distance, total_travel_time, worker_utilization,
            total_reward, total_quality, reward_quality_ratio, reward_distance_ratio, distance_quality_ratio)


def update_status(workers, tasks, individual, current_time):
    for task_id, worker_ids in individual.items():
        task = get_task_by_id(task_id, tasks)
        if worker_ids:
            for worker_id in worker_ids:
                worker = get_worker_by_id(worker_id, workers)

                # 计算任务开始时间
                start_time = calculate_worker_start_time(worker, current_time)
                # 计算任务完成时间
                travel_time = calculate_travel_time(worker, task)
                sensing_time = calculate_pair_sensing_time(worker, task)
                total_time = travel_time + sensing_time
                expected_finish_time = start_time + timedelta(hours=total_time)
                if task.task_type in ['TypeA', 'TypeB']:
                    task_location = task.location
                else:
                    task_location = calculate_nearest_location(worker, task)
                # 更新工人
                worker.last_task_location = task_location  # ***,一定要，后续计算距离省事
                reward = calculate_pair_sensing_bid(worker, task)
                quality = calculate_pair_sensing_quality(worker, task)

                # 记录任务信息到工人的assigned_tasks属性中
                task_info = {
                    'task_id': task.id,
                    'task_location': task_location,
                    'start_task': start_time,
                    'travel_time': travel_time,
                    'sensing_time': sensing_time,
                    'expected_finish_time': expected_finish_time,
                    'reward': reward,
                    'quality': quality
                }
                worker.assigned_tasks.append(task_info)

                # 更新任务
                worker_info = {
                    'worker_id': worker.id,
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


def calculate_avg_results(results_list):
    """计算指标平均值"""
    if not results_list:
        return []

    if isinstance(results_list[0], (list, tuple)):
        num_experiments = len(results_list)
        num_metrics = len(results_list[0])
        avg_results = [0] * num_metrics

        for result in results_list:
            for i in range(num_metrics):
                avg_results[i] += result[i]

        avg_results = [res / num_experiments for res in avg_results]
    else:
        avg_results = results_list

    return avg_results


def format_result_dict(num_workers, num_tasks, result_tuple, exp_id=None):
    """格式化结果为字典，用于写入 CSV"""
    res_dict = {
        'num_workers': num_workers,
        'num_tasks': num_tasks,
        'success_task_rate': f"{result_tuple[0]:.3f}",
        'total_distance': f"{result_tuple[1]:.3f}",
        'total_travel_time': f"{result_tuple[2]:.3f}",
        'worker_utilization': f"{result_tuple[3]:.3f}",
        'total_reward': f"{result_tuple[4]:.2f}",
        'total_quality': f"{result_tuple[5]:.2f}",
        'reward_quality_ratio': f"{result_tuple[6]:.3f}",
        'reward_distance_ratio': f"{result_tuple[7]:.3f}",
        'distance_quality_ratio': f"{result_tuple[8]:.3f}"
    }
    # 如果是详细数据，增加实验ID字段
    if exp_id is not None:
        res_dict['experiment_id'] = exp_id
    return res_dict


# --- 主逻辑函数 ---

def run_experiment_logic(mode, num_experiments, variable_list, constant_val,
                         time_slot, task_mode, ga_params, algorithms_to_run, output_dir):
    """
    通用的实验运行逻辑
    mode: 'task_fixed' (任务数固定) 或 'worker_fixed' (工人数固定)
    """

    # 字段定义
    base_fields = ['num_workers', 'num_tasks', 'success_task_rate', 'total_distance',
                   'total_travel_time', 'worker_utilization', 'total_reward', 'total_quality',
                   'reward_quality_ratio', 'reward_distance_ratio', 'distance_quality_ratio']

    detailed_fields = ['experiment_id'] + base_fields  # 详细文件多一个实验ID列

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 解析遗传算法参数
    num_gens, pop_size, mut_rate, tourn_size = ga_params

    # --- 针对每个算法进行循环 ---
    for algo_name in algorithms_to_run:
        suffix = "task" if mode == 'task_fixed' else "worker"
        print(f"\n{'=' * 20} 正在运行算法: {algo_name} ({mode}) {'=' * 20}")

        # 定义两个文件名：详细数据 和 平均数据
        file_detailed = os.path.join(output_dir, f'{algo_name}_{task_mode}_{suffix}_detailed.csv')
        file_avg = os.path.join(output_dir, f'{algo_name}_{task_mode}_{suffix}_avg.csv')

        # 同时打开两个文件
        with open(file_detailed, mode='w', newline='') as f_det, \
                open(file_avg, mode='w', newline='') as f_avg:

            writer_det = csv.DictWriter(f_det, fieldnames=detailed_fields)
            writer_avg = csv.DictWriter(f_avg, fieldnames=base_fields)

            writer_det.writeheader()
            writer_avg.writeheader()

            # 遍历变量列表 (工人数列表 或 任务数列表)
            for var_val in variable_list:
                # 确定当前的工人数和任务数
                if mode == 'task_fixed':
                    curr_workers = var_val
                    curr_tasks = constant_val
                else:
                    curr_workers = constant_val
                    curr_tasks = var_val

                print(f"\n--- 当前设置: 工人={curr_workers}, 任务={curr_tasks} ---")

                results_buffer = []

                # 执行多次实验
                for i in range(num_experiments):
                    print(f"  > 实验 {i + 1}/{num_experiments} running...", end="", flush=True)

                    # === 算法选择与调度 ===
                    if algo_name == 'GR':  # Greedy
                        res = run_greedy(curr_workers, curr_tasks, task_mode, time_slot)
                    elif algo_name == 'MFTA':  # Max Flow
                        res = run_maximum_flow(curr_workers, curr_tasks, task_mode, time_slot)
                    elif algo_name == 'GGA-I':  # Greedy Genetic
                        res = run_greedy_genetic(num_gens, pop_size, mut_rate,
                                                 curr_workers, curr_tasks, task_mode, time_slot)
                    elif algo_name == 'BPGA':  # My Genetic
                        res = run_my_genetic(num_gens, pop_size, mut_rate, tourn_size,
                                             curr_workers, curr_tasks, task_mode, time_slot)
                    else:
                        raise ValueError(f"未定义的算法名称: {algo_name}")

                    # 1. 实时打印结果
                    print(f" 完成. 结果: {res}")

                    # 2. 写入详细文件 (带实验ID)
                    row_data = format_result_dict(curr_workers, curr_tasks, res, exp_id=i + 1)
                    writer_det.writerow(row_data)
                    f_det.flush()  # 确保立即写入磁盘，防止程序中断数据丢失

                    results_buffer.append(res)

                # 计算平均值
                avg_results = calculate_avg_results(results_buffer)

                # 3. 写入平均值文件
                avg_row_data = format_result_dict(curr_workers, curr_tasks, avg_results)
                writer_avg.writerow(avg_row_data)
                f_avg.flush()

                print(f"  >>> 平均值已保存. 成功率: {avg_results[0]:.3f}")


def main_task_fixed():  # 任务数量固定总运行
    # 配置
    num_experiments = 10
    num_workers_list = [700, 750, 800, 850, 900, 950, 1000]
    num_tasks = 100
    time_slot = 30
    task_mode = 'uniform'

    # 遗传算法参数 (gen, pop, mut, tourn)
    ga_params = (100, 50, 0.1, 3)

    # 指定要运行的算法
    algorithms_to_run = ['GR', 'MFTA', 'GGA-I', 'BPGA']

    # 结果路径
    output_dir = os.path.join('random_bid_results', 'results1')

    # 调用通用逻辑
    run_experiment_logic(
        mode='task_fixed',
        num_experiments=num_experiments,
        variable_list=num_workers_list,  # 变量是工人数
        constant_val=num_tasks,  # 常量是任务数
        time_slot=time_slot,
        task_mode=task_mode,
        ga_params=ga_params,
        algorithms_to_run=algorithms_to_run,
        output_dir=output_dir
    )


def main_worker_fixed():  # 工人数量固定总运行
    # 配置
    num_experiments = 10
    num_workers = 1000
    num_tasks_list = [60, 80, 100, 120, 140, 160]
    time_slot = 30
    task_mode = 'uniform'

    # 遗传算法参数
    ga_params = (100, 50, 0.1, 3)

    # 指定要运行的算法
    algorithms_to_run = ['GR', 'MFTA', 'GGA-I', 'BPGA']

    # 结果路径
    output_dir = os.path.join('random_bid_results', 'results2')

    # 调用通用逻辑
    run_experiment_logic(
        mode='worker_fixed',
        num_experiments=num_experiments,
        variable_list=num_tasks_list,  # 变量是任务数
        constant_val=num_workers,  # 常量是工人数
        time_slot=time_slot,
        task_mode=task_mode,
        ga_params=ga_params,
        algorithms_to_run=algorithms_to_run,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # 运行
    print(">>> 启动任务数量固定实验...")
    main_task_fixed()

    print("\n>>> 启动工人数量固定实验...")
    main_worker_fixed()
