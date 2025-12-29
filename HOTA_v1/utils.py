"""
1.计算距离
2.计算移动时间
3.计算任务执行的操作时间
4.获取工人开始执行被分配的任务的时间
5.根据工人id获取工人，根据任务id获取任务
6.更新工人和用户状态

"""
import math
import random
from datetime import datetime as dt, time, timedelta

import numpy as np


# 计算曼哈顿距离
def manhattan_distance(location1, location2):
    return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])


# 计算工人执行任务的感知时间
def calculate_pair_sensing_time(worker, task):
    task_sensing_time = task.sensing_time / 60
    worker_proficiency = worker.proficiency.get(task.task_type, 0.5)

    if worker_proficiency <= 0.5:
        # 低熟练度情况，生成在 [0.9 * task_sensing_time, 1.5 * task_sensing_time] 范围内的高斯分布
        mean = 1.1 * task_sensing_time  # 均值设置为中间值
        std_dev = 0.1 * task_sensing_time  # 标准差可调整
        estimated_sensing_time = np.clip(np.random.normal(mean, std_dev),
                                         0.9 * task_sensing_time,
                                         1.5 * task_sensing_time)
    else:
        # 高熟练度情况，生成在 [0.75 * task_sensing_time, 1.2 * task_sensing_time] 范围内的高斯分布
        mean = (1 - ((worker_proficiency - 0.6) / 2)) * task_sensing_time  # 均值设置为中间值(0.8,1.05)
        std_dev = 0.05 * task_sensing_time  # 标准差可调整
        estimated_sensing_time = np.clip(np.random.normal(mean, std_dev),
                                         0.75 * task_sensing_time,
                                         1.2 * task_sensing_time)

    # 返回的时间四舍五入
    return round(estimated_sensing_time, 2)


def sigmoid(x, k=15, mid_point=0.65):
    return 1 / (1 + np.exp(-k * (x - mid_point)))


# 计算工人的任务的感知质量
def calculate_pair_sensing_quality(worker, task):
    min_quality = 0.4
    max_quality = 1.0

    worker_skill_level = worker.skill_level
    worker_proficiency = worker.proficiency.get(task.task_type, 0.5)
    # 计算技能水平的影响，使用sigmoid函数
    skill_quality = min_quality + (max_quality - min_quality) * sigmoid(worker_skill_level)
    # 计算熟练度的影响，使用sigmoid函数
    proficiency_impact = (max_quality - min_quality) * sigmoid(worker_proficiency, k=5, mid_point=0.75) - (
                max_quality - min_quality) / 2

    # 计算最终质量
    quality = skill_quality + proficiency_impact + 0.001
    # 保证质量范围在 [0.4, 1.0] 内
    sensing_quality = max(min_quality, min(max_quality, quality))

    return round(sensing_quality, 2)


def calculate_pair_sensing_bid(worker, task):
    # worker_skill_level = worker.skill_level
    task_type = task.task_type
    complexity = task.complexity
    task_type_score = get_score_for_type(task.task_type)
    # # 生成高斯扰动
    # X = round(np.clip(np.random.normal(0, 0.2), -0.5, 1), 2)
    # unit_sensing_bid = (5 / 3) * (worker_skill_level-0.3) + 1 + X     # 线性
    unit_sensing_bid = worker.unit_sensing_bid
    # sensing_bid = task_sensing_time * unit_sensing_bid * task_type_score * (1 + (task.complexity - 1) * 0.2)
    if task_type in ['TypeA', 'TypeB']:
        sensing_bid = complexity * unit_sensing_bid * task_type_score * 5
    else:
        sensing_bid = complexity * unit_sensing_bid * task_type_score * 10

    return round(sensing_bid, 2)


# 计算执行任务的时间
def calculate_worker_task_time(worker):
    total_task_time = timedelta()

    for task_info in worker.assigned_tasks:
        # 获取任务信息
        start_task = task_info['start_task']
        expected_finish_time = task_info['expected_finish_time']

        # 计算任务的时间跨度
        task_time = expected_finish_time - start_task
        # 累加任务的时间跨度
        total_task_time += max(task_time, timedelta())

    return total_task_time


def get_score_for_type(task_type):
    # 定义类型到分数的映射A:周期性多轮次任务;B:实时性多轮次任务;C:实时性数据采集任务;D:非实时数据采集任务
    type_to_score = {'TypeA': 1, 'TypeB': 1.2, 'TypeC': 1.4, 'TypeD': 1.6}

    # 获取并返回对应的分数，如果类型不在映射中则返回 None
    return type_to_score.get(task_type, None)


def calculate_worker_start_time(worker, current_time):
    if worker.assigned_tasks:
        last_task = worker.assigned_tasks[-1]
        task_completion_time = last_task['expected_finish_time']
        start_time = task_completion_time if task_completion_time > current_time else current_time
    else:
        start_time = current_time

    return start_time


def calculate_nearest_location(worker, task):
    if task.task_type in ['TypeA', 'TypeB']:
        return task.location

    min_distance = float('inf')
    nearest_location = task.location[0]  # 默认取第一个

    for location in task.location:
        distance = manhattan_distance(worker.location, location)
        if distance < min_distance:
            min_distance = distance
            nearest_location = location

    return nearest_location


# 计算工人到任务的距离
def calculate_distance(worker, task):
    # 检查工人是否有最后任务的位置，如果没有，使用当前时间获取位置
    if worker.last_task_location:
        worker_location = worker.last_task_location
    else:
        worker_location = worker.location
    distance = manhattan_distance(worker_location, calculate_nearest_location(worker, task))
    return round(distance, 3)


# 计算移动时间
def calculate_travel_time(worker, task):  # 单位是小时
    distance = calculate_distance(worker, task)
    # print(f"工人位置{worker_location}，任务距离{calculate_nearest_location(worker, task)}，距离为{distance}，工人速度为{worker.speed}")
    travel_time = distance / worker.speed
    return round(travel_time, 3)


# 计算执行任务的操作时间
def calculate_execute_time(worker, task):  # 单位是小时
    task_type_score = get_score_for_type(task.task_type)
    proficiency = worker.proficiency.get(task_type_score, 0)
    fatigue_factor = math.log(worker.fatigue + 1) + 1
    execute_time = task.sensing_time / 60 * (1 - 0.25 * (proficiency - 0.5)) * fatigue_factor
    execute_time = round(execute_time, 2)
    return round(execute_time, 3)


def calculate_start_time(worker, current_time):
    if worker.assigned_tasks:
        last_task = worker.assigned_tasks[-1]
        task_completion_time = last_task['expected_finish_time']
        start_time = task_completion_time if task_completion_time > current_time else current_time
    else:
        start_time = current_time

    return start_time


def to_dt(time):
    return dt.combine(dt.today(), time)


def get_worker_by_id(worker_id, workers):
    for worker in workers:
        if worker.id == worker_id:
            return worker
    return None  # 如果找不到相应的工人，则返回 None


def get_task_by_id(task_id, tasks):
    for task in tasks:
        if task.id == task_id:
            return task
    return None  # 如果找不到相应的任务，则返回 None


# 更新状态
def update_status(worker, task, start_task, travel_time, execute_time, expected_finish_time):
    # 更新工人
    worker.last_task_location = task.location
    # 更新疲劳度，每移动一公里或每小时任务执行，疲劳度相应增长
    additional_fatigue = math.exp(0.5 * travel_time) + math.exp(execute_time)
    worker.fatigue += additional_fatigue
    # 记录任务信息到工人的assigned_tasks属性中
    task_info = {
        'task_id': task.tid,
        'task_location': task.location,
        'start_task': start_task,
        'travel_time': travel_time,
        'execute_time': execute_time,
        'expected_finish_time': expected_finish_time
    }
    worker.assigned_tasks.append(task_info)

    # 更新任务
    worker_info = {
        'worker_id': worker.wid,
        'start_task': start_task,
        'travel_time': travel_time,
        'execute_time': execute_time,
        'expected_finish_time': expected_finish_time
    }
    task.assigned_workers.append(worker_info)
    return 0
