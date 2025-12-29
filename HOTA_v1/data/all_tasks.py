"""
生成不同类型的任务，计划先生成，再按时间排序，给id赋值
"""
import numpy as np

from utils import to_dt
import random
from datetime import datetime as dt, timedelta, time


class Task:
    def __init__(self, tid, task_type, complexity, quality_threshold, location, num_area, start_time, end_time,
                 budget, sensing_time, sensing_rounds, interval):
        self.id = tid
        self.task_type = task_type
        self.complexity = complexity
        self.quality_threshold = quality_threshold
        self.location = location
        self.num_area = num_area
        self.start_time = start_time
        self.end_time = end_time
        self.budget = budget
        self.sensing_time = sensing_time
        self.sensing_rounds = sensing_rounds
        self.interval = interval

        self.assigned_workers = []

    def is_to_assign(self, current_time):
        if to_dt(self.start_time) <= current_time < to_dt(self.end_time):
            if self.task_type in ['TypeA', 'TypeB']:
                if self.sensing_rounds == len(self.assigned_workers):
                    return False
                else:
                    return True
            else:
                if len(self.assigned_workers) < self.num_area:
                    return True
                else:
                    return False
        else:
            return False

    def to_dict(self):
        return {
            'id': self.id,
            'task_type': self.task_type,
            'complexity': self.complexity,
            'quality_threshold': self.quality_threshold,
            'location': self.location,
            'num_area': self.num_area,
            'start_time': self.start_time.strftime('%H:%M:%S'),
            'end_time': self.end_time.strftime('%H:%M:%S'),
            'budget': self.budget,
            'sensing_time': self.sensing_time,
            'sensing_rounds': self.sensing_rounds,
            'interval': self.interval,
            'assigned_workers': self.assigned_workers
        }

    @classmethod
    def from_dict(cls, task_dict):
        def parse_time(t_str):
            return dt.strptime(t_str, '%H:%M:%S').time()

        start_time = parse_time(task_dict.get('start_time'))
        end_time = parse_time(task_dict.get('end_time'))
        num_area = task_dict.get('num_area', 0)
        task = cls(
            task_dict.get('id'),
            task_dict.get('task_type'),
            task_dict.get('complexity'),
            task_dict.get('quality_threshold'),
            task_dict.get('location'),
            num_area,
            start_time,
            end_time,
            task_dict.get('budget'),
            task_dict.get('sensing_time'),
            task_dict.get('sensing_rounds'),
            task_dict.get('interval')
        )
        task.assigned_workers = task_dict.get('assigned_workers', [])
        return task


def location_mode(mode='uniform'):
    location = [0, 0]
    # 生成位置根据不同模式调整
    if mode == 'uniform':  # 均匀分布
        # location = (random.randint(5, 45), random.randint(5, 45))
        location = (random.randint(0, 50), random.randint(0, 50))
    elif mode == 'concentrated':  # 集中分布
        # 定义五个固定区域的范围
        regions = [(24, 24, 29, 34), (39, 11, 44, 21), (1, 21, 11, 26), (36, 40, 46, 45), (11, 37, 16, 47)]
        # 随机选择一个区域
        selected_region_index = random.randint(0, 4)
        selected_region = regions[selected_region_index]

        # 在选定的区域内生成一个随机点
        random_point_x = random.randint(selected_region[0], selected_region[2])
        random_point_y = random.randint(selected_region[1], selected_region[3])
        location = (random_point_x, random_point_y)
    elif mode == 'mixed':  # 混合式分布
        if random.random() < 0.5:  # 50%概率均匀分布
            # location = (random.randint(5, 45), random.randint(5, 45))
            location = (random.randint(0, 50), random.randint(0, 50))
        else:  # 50%概率集中分布
            # 定义五个固定区域的范围
            regions = [(24, 24, 29, 34), (39, 11, 44, 21), (1, 21, 11, 26), (36, 40, 46, 45), (11, 37, 16, 47)]
            # 随机选择一个区域
            selected_region_index = random.randint(0, 4)
            selected_region = regions[selected_region_index]
            # 在选定的区域内生成一个随机点
            random_point_x = random.randint(selected_region[0], selected_region[2])
            random_point_y = random.randint(selected_region[1], selected_region[3])
            location = (random_point_x, random_point_y)
    return location


def generate_continuous_positions(start_location, x_extent, y_extent):  # 模运算处理边界超限

    positions = []
    step = 0.5
    start_x, start_y = start_location
    x_bound, y_bound = 50, 50
    for i in range(x_extent):
        for j in range(y_extent):
            x_pos = start_x + i * step
            y_pos = start_y + j * step

            # 处理超出边界的情况
            if x_pos > x_bound:
                x_pos = x_bound - i * step
            if y_pos > x_bound:
                y_pos = y_bound - j * step

            positions.append((x_pos, y_pos))

    return positions


def generate_tasks(num_tasks, mode, type_ratios):
    task_types = list(type_ratios.keys())
    tasks = []

    for i in range(num_tasks):
        task_type = random.choices(task_types, weights=type_ratios.values())[0]  # 随机选择一个任务类型（有权重）
        # complexity = random.choice([1, 2, 3])
        # quality_threshold = 0.6 + 0.1 * (complexity - 1)
        complexity, quality_threshold = 0, 0
        location = [0, 0]
        num_area = 1
        start_time, end_time = time(0, 0, 0), time(21, 0, 0)
        budget, sensing_rounds, sensing_time, interval = 0, 0, 0, 0

        if task_type == 'TypeA':  # 简单周期性多轮次任务
            location = location_mode(mode)
            complexity = random.choice([1, 2])
            quality_threshold = 0.6 + 0.1 * (complexity - 1)
            sensing_rounds = random.choices([4, 5, 6], weights=[3, 2, 1], k=1)[0]
            sensing_time = 5 * complexity
            interval = random.choice([0.75, 1])  # 45min/1h
            duration = sensing_rounds * interval
            latest_start_hour = int(20 - duration)
            start_hour = random.randint(8, latest_start_hour - 2) if latest_start_hour > 8 else 8
            start_minute = random.choice([0, 30])
            start_time = time(start_hour, start_minute, 0)
            # 将 start_time 转换为 datetime 对象
            start_datetime = dt.combine(dt.today(), start_time)
            # 计算结束时间
            end_datetime = start_datetime + timedelta(hours=duration)
            if end_datetime > dt.combine(dt.today(), time(20, 0, 0)):
                end_time = time(20, 0, 0)
            else:
                end_time = end_datetime.time()
            budget = (20 + 2 * sensing_time) * sensing_rounds

        elif task_type == 'TypeB':  # 实时性多轮次任务
            location = location_mode(mode)
            complexity = random.choice([2, 3])
            quality_threshold = 0.6 + 0.1 * (complexity - 1)
            sensing_rounds = random.randint(1, 4)
            sensing_time = 5 * complexity
            interval = random.choice([0.75, 1])  # 45min/1h
            duration = sensing_rounds * interval
            latest_start_hour = int(20 - duration)
            start_hour = random.randint(8, latest_start_hour - 2) if latest_start_hour > 8 else 8
            start_minute = random.choice([0, 30])
            start_time = time(start_hour, start_minute, 0)
            # 将 start_time 转换为 datetime 对象
            start_datetime = dt.combine(dt.today(), start_time)
            # 计算结束时间
            end_datetime = start_datetime + timedelta(hours=duration)
            if end_datetime > dt.combine(dt.today(), time(20, 0, 0)):
                end_time = time(20, 0, 0)
            else:
                end_time = end_datetime.time()
            budget = (20 + 2 * sensing_time) * sensing_rounds

        elif task_type == 'TypeC':  # 非实时性数据采集任务
            complexity = random.choice([1, 2, 3])
            quality_threshold = 0.6 + 0.1 * (complexity - 1) + 0.05
            location0 = location_mode(mode)
            # 随机选择区域扩展尺寸（2x3, 3x3, 3x4）
            possible_extents = [(2, 3), (3, 3), (3, 4)]
            x_extent, y_extent = random.choice(possible_extents)
            num_area = x_extent * y_extent
            location = generate_continuous_positions(location0, x_extent, y_extent)
            sensing_time = 10 * complexity
            duration = random.randint(4, 6)
            latest_start_hour = int(20 - duration)
            start_hour = random.randint(8, latest_start_hour - 1) if latest_start_hour > 8 else 8
            start_time = time(start_hour, random.randint(0, 59), random.randint(0, 59))
            end_datetime = to_dt(start_time) + timedelta(hours=duration)
            if end_datetime > dt.combine(dt.today(), time(20, 0, 0)):
                end_time = time(20, 0, 0)
            else:
                end_time = end_datetime.time()
            budget = (40 + 2 * sensing_time) * num_area

        elif task_type == 'TypeD':  # 实时数据采集任务
            complexity = random.choice([1, 2])
            quality_threshold = 0.6 + 0.1 * (complexity - 1) + 0.05
            location0 = location_mode(mode)
            possible_extents = [(1, 3), (2, 3), (3, 3)]
            x_extent, y_extent = random.choice(possible_extents)
            num_area = x_extent * y_extent
            location = generate_continuous_positions(location0, x_extent, y_extent)
            sensing_time = 10 * complexity
            duration = random.randint(2, 4)  # 2/3/4h
            latest_start_hour = int(20 - duration)
            start_hour = random.randint(8, latest_start_hour - 1) if latest_start_hour > 8 else 8
            start_time = time(start_hour, random.randint(0, 59), random.randint(0, 59))
            end_datetime = to_dt(start_time) + timedelta(hours=duration)
            if end_datetime > dt.combine(dt.today(), time(20, 0, 0)):
                end_time = time(20, 0, 0)
            else:
                end_time = end_datetime.time()
            budget = (30 + 2 * sensing_time) * num_area

        task = Task(i + 1, task_type, complexity, quality_threshold, location, num_area, start_time, end_time, budget,
                    sensing_time, sensing_rounds, interval)
        tasks.append(task)

    return tasks

# # 生成任务示例
# num_tasks = 300
# mode = 'uniform'
# type_ratios = {'TypeA': 0.3, 'TypeB': 0.2, 'TypeC': 0.4, 'TypeD': 0.1}
# tasks = generate_tasks(num_tasks, mode, type_ratios)
#
# for task in tasks:
#     print(task.__dict__)
