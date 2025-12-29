'''
生成工人，考虑信誉度
'''
import math
import random
from datetime import datetime as dt, timedelta, time

import numpy as np

from utils import to_dt


class Worker:
    def __init__(self, wid, location, speed, start_time, end_time, reputation, skill_level, proficiency,
                 unit_sensing_bid, is_malicious):
        self.id = wid
        self.location = location
        self.speed = speed
        self.start_time = start_time
        self.end_time = end_time
        self.reputation = reputation
        self.skill_level = skill_level
        self.proficiency = proficiency
        self.unit_sensing_bid = unit_sensing_bid
        self.is_malicious = is_malicious

        self.last_task_location = None
        self.assigned_tasks = []

    # 判断工人是否可用：1.在工作时间；2.在当前时间后十分钟之内有空
    def is_available(self, current_time):
        if to_dt(self.start_time) <= current_time < to_dt(self.end_time):
            if not self.last_task_location:
                return True
            else:
                last_task = self.assigned_tasks[-1]
                return last_task['expected_finish_time'] <= current_time + timedelta(minutes=10)
        else:
            return False

    def to_dict(self):
        return {
            'id': self.id,
            'location': self.location,
            'speed': self.speed,
            'start_time': self.start_time.strftime('%H:%M:%S'),
            'end_time': self.end_time.strftime('%H:%M:%S'),
            'reputation': self.reputation,
            'skill_level': self.skill_level,
            'proficiency': self.proficiency,
            'unit_sensing_bid': self.unit_sensing_bid,
            'is_malicious': self.is_malicious,
            'last_task_location': self.last_task_location,
            'assigned_tasks': self.assigned_tasks
        }

    @classmethod
    def from_dict(cls, worker_dict):
        def parse_time(t_str):
            return dt.strptime(t_str, '%H:%M:%S').time()

        start_time = parse_time(worker_dict['start_time'])
        end_time = parse_time(worker_dict['end_time'])
        worker = cls(
            worker_dict['id'],
            worker_dict['location'],
            worker_dict['speed'],
            start_time,
            end_time,
            worker_dict['reputation'],
            worker_dict['skill_level'],
            worker_dict['proficiency'],
            worker_dict['unit_sensing_bid'],
            worker_dict['is_malicious']
        )
        worker.last_task_location = worker_dict.get('last_task_location', None)
        worker.assigned_tasks = worker_dict.get('assigned_tasks', [])
        return worker


def generate_workers(num_workers):
    workers = []
    is_malicious = random.random() < 0.15   # 恶意用户的比例设置为 15%，生成一个在 [0.0, 1.0) 范围内的随机浮点数
    task_type_list = ['TypeA', 'TypeB', 'TypeC', 'TypeD']
    for i in range(num_workers):
        location = (round(random.uniform(0, 50), 1), round(random.uniform(0, 50), 1))
        speed = round(random.uniform(7, 10), 2) * 2

        # 生成随机的小时、分钟和秒
        start_time = time(random.randint(7, 18), random.randint(0, 59), random.randint(0, 59))
        if to_dt(start_time) < to_dt(time(8, 0, 0)):
            start_time = time(8, 0, 0)

        # 根据高斯分布生成可用时间，大部分为2.5-4.5，不小于1h，不超过6h
        available_time = round(np.clip(random.gauss(3.5, 1.0), 1.0, 6.0), 1)

        end_time = to_dt(start_time) + timedelta(hours=available_time)
        # 检查结束时间是否晚于8:00:00
        latest_time = to_dt(time(20, 0, 0))
        if end_time > latest_time:
            end_time = latest_time

        end_time = dt.time(end_time)

        prob = random.random()      # 20%的新用户
        if prob < 0.2:
            reputation = 0.5
        elif is_malicious:  # 恶意用户，通过历史数据已知
            reputation = round(random.uniform(0.3, 0.5), 2)
        else:   # 非恶意用户
            reputation = round(np.clip(np.random.normal(0.7, 0.1), 0.5, 0.95), 2)

        if reputation == 0.5:
            skill_level = 0.5
        else:
            skill_level = round(np.clip(np.random.normal(0.7, 0.1), 0.5, 1.0), 2)

        proficiency = {}
        for task_type in task_type_list:
            if reputation == 0.5:
                task_proficiency = 0.5
            else:
                task_proficiency = round(np.clip(np.random.normal(0.7, 0.15), 0.5, 1.0), 2)
            proficiency[task_type] = task_proficiency

        # unit_sensing_bid = round(random.uniform(1, 3), 2)

        # 生成高斯扰动
        X = round(np.clip(np.random.normal(0, 0.2), -0.2, 0.5), 2)
        # unit_sensing_bid = (5 / 3) * (skill_level - 0.5) + 1 + X  # 线性
        # unit_sensing_bid = 1 + (skill_level - 0.4) / 0.6 + X  # 线性
        unit_sensing_bid = 2 * math.log(2 * skill_level + 1) + X

        worker = Worker(i + 1, location, speed, start_time, end_time, reputation, skill_level,
                        proficiency, unit_sensing_bid, is_malicious)
        workers.append(worker)

    return workers


# num_workers = 300
# workers = generate_workers(num_workers)
#
# for worker in workers:
#     print(worker.__dict__)
