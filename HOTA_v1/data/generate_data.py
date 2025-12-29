"""
随机随机生成500个工人和300个任务
"""
import json
import os

from matplotlib import pyplot as plt
from data.all_workers import *
from data.all_tasks import *


# 定义存储数据的文件夹路径
DATA_FOLDER = 'new_data'


def save_to_file(filename, data):
    file_path = os.path.join(DATA_FOLDER, filename)  # 构建完整的文件路径
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def generate_workers_data(num_workers):
    workers = generate_workers(num_workers)

    # 将工人和任务转换为字典列表
    workers_data = [worker.to_dict() for worker in workers]

    # 保存到文件
    save_to_file('workers_data.json', workers_data)


def generate_tasks_data(num_tasks, task_mode, type_ratios):
    def extract_location(task):
        # 如果位置是列表，取第一个位置；否则，直接使用位置
        location = task.location
        if isinstance(location, list) and location:
            return location[0]
        return location

    if task_mode == "uniform":
        tasks = generate_tasks(num_tasks, "uniform", type_ratios)
    elif task_mode == "concentrated":
        tasks = generate_tasks(num_tasks, "concentrated", type_ratios)
    elif task_mode == "mixed":
        tasks = generate_tasks(num_tasks, "mixed", type_ratios)
    else:
        print("模式输入错误！")
        return

    tasks_data = [task.to_dict() for task in tasks]
    save_to_file(f'{task_mode}_tasks_data.json', tasks_data)

    locations = [extract_location(task) for task in tasks]
    if not locations:
        print("没有任务位置数据可供绘制。")
    else:
        x, y = zip(*locations)
        plt.scatter(x, y)
        # plt.title(f'{task_mode.capitalize()} distributed random task locations')
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)

        # 保存图像，增加DPI以提高清晰度
        plt.savefig(os.path.join(DATA_FOLDER, f'{task_mode}_task_locations.svg'), dpi=600)  # 或使用 .svg 格式
        plt.show()


if __name__ == "__main__":
    num_workers = 1200
    num_tasks = 200
    # task_mode = 'concentrated'   # uniform, concentrated, mixed
    task_mode_list = {'uniform', 'concentrated', 'mixed'}
    type_ratios = {'TypeA': 0.3, 'TypeB': 0.2, 'TypeC': 0.3, 'TypeD': 0.2}

    # 确保存储数据的文件夹存在
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    print(f"开始生成{DATA_FOLDER}数据")

    generate_workers_data(num_workers)
    print(f"{num_workers}个工人随机数据生成完成！")
    for task_mode in task_mode_list:
        generate_tasks_data(num_tasks, task_mode, type_ratios)
        print(f"{num_tasks}个{task_mode}任务随机数据生成完成！")
    # generate_tasks_data(num_tasks, task_mode, type_ratios)      # 生成单个类型的数据
    # print(f"{task_mode}任务随机数据生成完成！")
