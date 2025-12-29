import matplotlib.pyplot as plt
import pandas as pd
import os

# 设置全局字体大小
plt.rcParams.update({'font.size': 22})  # 设定全局字体大小


def plot_algorithm_comparison(folder_path, algorithms, task_mode, y_column_name, save_path):
    plt.figure(figsize=(10, 6))  # 设置图表大小

    ylabel = ""  # 初始化 ylabel 变量

    # 设置柱状图的宽度
    bar_width = 0.2
    index = 0
    x_values = []

    # 颜色列表：每个算法对应一种颜色
    colors = ['b', 'orange', 'g', 'r']

    for i, algorithm in enumerate(algorithms):
        file_name = f"{algorithm}_{task_mode}_task_results.csv"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            x_values = df['num_workers']
            y_values = df[y_column_name]

            # 检查 y_column_name 是否为 'worker_utilization'
            if y_column_name == 'worker_utilization':
                ylabel = "participant utilization"
            else:
                ylabel = y_column_name.replace('_', ' ')

            x_positions = [pos + bar_width * index for pos in range(len(x_values))]

            # 使用颜色列表中对应的颜色
            plt.bar(x_positions, y_values, width=bar_width, label=algorithm, color=colors[i])
            index += 1

    plt.xlabel('Number of participants', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    # plt.title(f'{task_mode} distribution', fontsize=24)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=18)  # 设置图例位置为右下角，字体较小
    plt.xticks([pos + bar_width * (len(algorithms) - 1) / 2 for pos in range(len(x_values))], x_values)
    plt.tight_layout()

    # 保存高清图像
    plt.savefig(save_path, dpi=1200)  # 设置DPI为1200，保存为PNG格式
    plt.close()  # 关闭图表以释放内存


def plot_algorithm_comparison2(folder_path, algorithms, task_mode, y_column_name, save_path):
    plt.figure(figsize=(10, 6))

    bar_width = 0.2
    index = 0
    x_values = []  # 在循环外定义 x_values

    # 颜色列表
    colors = ['b', 'orange', 'g', 'r']

    for algorithm in algorithms:
        file_name = f"{algorithm}_{task_mode}_worker_results.csv"
        file_path = os.path.join(folder_path, file_name)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            x_values = df['num_tasks']
            y_values = df[y_column_name]
            x_positions = [pos + bar_width * index for pos in range(len(x_values))]

            # 使用颜色列表中的颜色
            plt.bar(x_positions, y_values, width=bar_width, label=algorithm, color=colors[index])

            index += 1

    plt.xlabel('Number of tasks', fontsize=22)
    plt.ylabel(y_column_name, fontsize=22)
    # plt.title(f'{task_mode} distribution', fontsize=24)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=18)
    plt.xticks([pos + bar_width * (len(algorithms) - 1) / 2 for pos in range(len(x_values))], x_values)
    plt.tight_layout()

    plt.savefig(save_path, dpi=1200)  # 设置DPI为1200
    plt.close()  # 关闭图表以释放内存


# # # 文件夹路径
# results_folder = 'random_bid_results/results1'
# savefig_folder = 'fig'
# algorithms = ['GR', 'MFTA', 'GGA-I', 'BPGA']
# task_mode = 'mixed'  # uniform, concentrated, mixed
# metrics = ['total_distance', 'total_travel_time']
#
# for metric in metrics:
#     # save_path = f'{task_mode}_{metric}1.pdf'  # 指定保存为PNG格式的文件名
#     save_path = os.path.join(savefig_folder, f'{task_mode}_{metric}.pdf')  # 保存路径
#     plot_algorithm_comparison(results_folder, algorithms, task_mode, metric, save_path)


# 文件夹路径
results_folder = 'random_bid_results/results2'
savefig_folder = 'fig2'
algorithms = ['GR', 'MFTA', 'GGA-I', 'BPGA']
task_mode = 'concentrated'  # uniform, concentrated, mixed
metrics = ['total_distance', 'total_travel_time']


for metric in metrics:
    # save_path = f'{task_mode}_{metric}1.pdf'  # 指定保存为PNG格式的文件名
    save_path = os.path.join(savefig_folder, f'{task_mode}_{metric}2.pdf')  # 保存路径
    plot_algorithm_comparison2(results_folder, algorithms, task_mode, metric, save_path)
