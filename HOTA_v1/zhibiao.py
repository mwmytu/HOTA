"""
1.任务分配率（1.0）
2.移动距离
3.工人利用率
4.工人的平均任务个数
5.总的时间开销
6.总移动时间
7.总报酬
8。总质量

"""
from utils import manhattan_distance, calculate_worker_task_time, to_dt
from datetime import timedelta


def calculate_success_task_rate(tasks):
    success_task_count = 0
    for task in tasks:
        if task.task_type in ['TypeA', 'TypeB']:
            if len(task.assigned_workers) >= task.sensing_rounds:
                success_task_count += 1
        else:
            if len(task.assigned_workers) >= task.num_area:
                success_task_count += 1

    success_task_rate = success_task_count / len(tasks)
    return success_task_rate


# 计算移动总距离
def calculate_total_distance(workers):
    total_distance = 0

    for worker in workers:
        if not worker.assigned_tasks:
            continue
        previous_location = worker.location
        for task_info in worker.assigned_tasks:
            total_distance += manhattan_distance(previous_location, task_info['task_location'])
            previous_location = task_info['task_location']

    return round(total_distance, 3)  # 保留三位小数


# 计算工人的利用率--分配到工人的数量
def calculate_worker_utilization(workers):
    assigned_workers = 0
    for worker in workers:
        if worker.assigned_tasks:
            assigned_workers += 1
    worker_utilization = assigned_workers / len(workers)
    return round(worker_utilization, 3)


# 计算工人的平均子任务分配数量
def average_num_subtasks(workers):
    worker_utilization = calculate_worker_utilization(workers)
    assigned_sub_tasks = 0
    for worker in workers:
        assigned_sub_tasks += len(worker.assigned_tasks)
    avg_num_tasks = assigned_sub_tasks / (worker_utilization * len(workers))
    return round(avg_num_tasks, 3)


# 总的时间开销
def calculate_total_time_cost(tasks):
    total_time_cost = timedelta(minutes=0)
    success_task_rate = calculate_success_task_rate(tasks)
    for task in tasks:
        if task.assigned_workers:
            for worker_info in task.assigned_workers:
                total_time_cost += worker_info['expected_finish_time'] - worker_info['start_task']
    total_time_cost = total_time_cost / success_task_rate
    total_time_cost_in_seconds = int(total_time_cost.total_seconds())/60
    return round(total_time_cost_in_seconds, 3)


# 总移动时间
def calculate_total_travel_time(workers):
    total_travel_time = 0
    for worker in workers:
        if worker.assigned_tasks:
            for worker_info in worker.assigned_tasks:
                travel_time = worker_info['travel_time']
                total_travel_time += travel_time

    return round(total_travel_time, 3)


def calculate_total_reward(tasks):
    total_reward = 0
    for task in tasks:
        if task.assigned_workers:
            for task_info in task.assigned_workers:
                total_reward += task_info['reward']
    return round(total_reward, 2)


def calculate_total_quality(tasks):
    total_quality = 0.0
    for task in tasks:
        if task.assigned_workers:
            for task_info in task.assigned_workers:
                total_quality += task_info['quality']
    return round(total_quality, 2)
