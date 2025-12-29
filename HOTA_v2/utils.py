"""
1.计算距离
2.计算移动时间
3.计算任务执行的操作时间
4.获取工人开始执行被分配的任务的时间
5.根据工人id获取工人，根据任务id获取任务
6.更新工人和用户状态

"""
import math
from datetime import datetime as dt, timedelta
import numpy as np
import pandas as pd


# 用两个经纬度坐标计算曼哈顿距离，返回公里
def haversine(lat1, lon1, lat2, lon2):
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # 计算纬度和经度的差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # 使用地球半径计算距离
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 地球半径（公里）
    R = 6371.0

    # 返回弧长距离
    return R * c


def manhattan_distance(location1, location2):
    lat1, lon1 = location1
    lat2, lon2 = location2
    # 计算纬度和经度的差值
    dlat = abs(lat2 - lat1)
    dlon = abs(lon2 - lon1)

    # 将差值转换为公里数
    lat_distance = haversine(lat1, lon1, lat1 + dlat, lon1)
    lon_distance = haversine(lat1, lon1, lat1, lon1 + dlon)

    # 计算曼哈顿距离
    return lat_distance + lon_distance


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
    quality = skill_quality + proficiency_impact
    # 保证质量范围在 [0.4, 1.0] 内
    sensing_quality = max(min_quality, min(max_quality, quality))

    return round(sensing_quality, 2)


def calculate_pair_sensing_bid(worker, task):
    base_time = task.sensing_time / 60.0  # 转换为小时
    unit_price = worker.unit_sensing_bid
    complexity_factor = task.complexity
    type_score = get_score_for_type(task.task_type)

    # 公式：时长 * 单价 * 难度溢价 * 类型溢价
    sensing_bid = base_time * unit_price * complexity_factor * type_score

    return round(sensing_bid, 2)


# 计算总报酬=距离补偿+感知报价
def calculate_pair_sensing_reward(worker, task, current_time):
    bid_per_distance = 1.5
    distance = calculate_distance(worker, task, current_time)
    bid = calculate_pair_sensing_bid(worker, task)
    reward = bid_per_distance * distance + bid
    return reward


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


# 获取工人位置
def get_worker_location(worker, current_time):
    if worker.assigned_tasks:
        return worker.last_task_location

    # 如果工人没有分配任务或任务已完成，从轨迹文件中获取位置
    df = pd.read_csv(worker.trajectory_file, header=None,
                     names=["vehicle_id", "timestamp", "longitude", "latitude"])

    # 将轨迹文件中的时间转换为 time 对象（去掉日期）
    df['time'] = pd.to_datetime(df['timestamp']).dt.time

    # 将 time 转换为 datetime 对象
    df['datetime'] = df['time'].apply(add_date)

    # 过滤出早于当前时间的记录
    df_filtered = df[df['datetime'] <= current_time]

    if not df_filtered.empty:
        # 获取最近的一条记录
        latest_record = df_filtered.iloc[-1]
        return (latest_record['latitude'], latest_record['longitude'])
    else:
        # 如果没有符合条件的记录，返回工人的初始位置
        return worker.location


def calculate_nearest_location(worker, task, current_time):
    worker_location = get_worker_location(worker, current_time)
    if task.task_type in ['TypeA', 'TypeB']:
        return task.location

    min_distance = float('inf')
    nearest_location = task.location[0]  # 默认取第一个

    for location in task.location:
        distance = manhattan_distance(worker_location, location)
        if distance < min_distance:
            min_distance = distance
            nearest_location = location

    return nearest_location


# 计算工人到任务的距离
def calculate_distance(worker, task, current_time):
    # 检查工人是否有最后任务的位置，如果没有，使用当前时间获取位置
    worker_location = get_worker_location(worker, current_time)
    task_location = calculate_nearest_location(worker, task, current_time)
    distance = manhattan_distance(worker_location, task_location)
    # print(f"工人位置为{worker_location}，任务位置为{task_location}， 距离为：{distance}")
    return round(distance, 3)


# 计算移动时间
def calculate_travel_time(worker, task, current_time):  # 单位是小时
    distance = calculate_distance(worker, task, current_time)
    # print(f"距离为{distance}，工人速度为{worker.speed}")
    travel_time = distance / (worker.speed * 2)
    return round(travel_time, 3)


# 计算执行任务的操作时间
def calculate_execute_time(worker, task):  # 单位是小时
    proficiency = worker.proficiency.get(task.task_type, 0.5)
    fatigue_factor = math.log(worker.fatigue + 1) + 1
    execute_time = task.sensing_time / 60 * (1 - 0.25 * (proficiency - 0.5)) * fatigue_factor
    return round(execute_time, 3)


def calculate_start_time(worker, current_time):
    if worker.assigned_tasks:
        last_task = worker.assigned_tasks[-1]
        task_completion_time = last_task['expected_finish_time']
        start_time = task_completion_time if task_completion_time > current_time else current_time
    else:
        start_time = current_time

    return start_time


def add_date(time):
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
