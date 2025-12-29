"""
按报价进行贪心，低报酬优先
ps：1.按照任务剩余时间，时间短的优先分配；2.不对移动时间有限制；3.满足质量约束的才能分配
"""
from utils import *
import datetime


# 按报价进行贪心分配
def greedy_bid_assign(workers, tasks, current_time, time_slot):
    # 根据任务类型分别计算剩余时间并排序
    tasks.sort(key=lambda x:
        (to_dt(x.start_time) + timedelta(hours=(len(x.assigned_workers) + 1) * x.interval) - current_time).seconds
        if x.task_type in ['TypeA', 'TypeB']
        else (to_dt(x.end_time) - current_time).seconds)

    assignments = {task.id: [] for task in tasks}  # 初始化字典以存储分配结果

    available_workers = workers.copy()
    assigned_worker_ids = set()  # 用于跟踪已分配的工人ID，另加

    for task in tasks:
        candidate_workers = []
        for worker in available_workers:
            if worker.id in assigned_worker_ids:  # 如果工人已经被分配，则跳过
                continue

            if not worker.is_available(current_time):  # 如果工人当前不可用，跳过此工人
                continue

            if task.task_type in ['TypeA', 'TypeB']:
                task_start = to_dt(task.start_time) + timedelta(hours=task.interval * len(task.assigned_workers))
                worker_start = calculate_worker_start_time(worker, current_time)
                start_task = worker_start if worker_start > task_start else task_start
            else:
                start_task = calculate_worker_start_time(worker, current_time)

            # 计算到达并完成任务需要的时间
            travel_time = calculate_travel_time(worker, task)
            # 检查 travel_time 是否超过2个分配时间间隔
            # if travel_time > 2 * time_slot:
            #     continue
            if travel_time > time_slot:
                continue

            sensing_time = calculate_pair_sensing_time(worker, task)
            total_time = travel_time + sensing_time
            # print(f"worker{worker.id}执行{task.task_type}的任务{task.id}，移动时间{travel_time}、执行时间{sensing_time}")

            # 检查工人完成任务的时间是否在工人的工作时间和任务截止时间之内
            expected_finish_time = start_task + timedelta(hours=total_time)
            if task.task_type not in ['TypeA', 'TypeB']:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= to_dt(task.end_time)):
                    sensing_quality = calculate_pair_sensing_quality(worker, task)
                    if sensing_quality >= task.quality_threshold:
                        reward = calculate_pair_sensing_bid(worker, task)
                        candidate_workers.append((worker, travel_time, total_time, reward))
            else:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= (to_dt(task.start_time) + timedelta(
                            hours=(len(task.assigned_workers) + 1) * task.interval))):
                    sensing_quality = calculate_pair_sensing_quality(worker, task)
                    if sensing_quality >= task.quality_threshold:
                        reward = calculate_pair_sensing_bid(worker, task)
                        candidate_workers.append((worker, travel_time, reward, sensing_quality))

        # 按预计报酬进行排序
        candidate_workers.sort(key=lambda x: x[2])
        # print(f"{task.task_type}的{task.id}的候选工人数：{len(candidate_workers)}")

        # 只为任务分配一个合适的工人
        if candidate_workers:
            if task.task_type in ['TypeA', 'TypeB']:
                chosen_worker, _, _, _ = candidate_workers[0]
                assignments[task.id].append(chosen_worker.id)
                assigned_worker_ids.add(chosen_worker.id)  # 记录已分配的工人
                available_workers.remove(chosen_worker)
            else:
                # 计算剩余可分配的工人数量
                remaining_workers_count = (task.num_area - len(task.assigned_workers))
                # 选择分配工人数量为剩余可分配数量和候选工人数量的较小值
                num_workers_to_assign = min(remaining_workers_count, len(candidate_workers))

                for i in range(num_workers_to_assign):
                    chosen_worker, _, _, _ = candidate_workers[i]
                    assignments[task.id].append(chosen_worker.id)
                    assigned_worker_ids.add(chosen_worker.id)  # 记录已分配的工人
                    available_workers.remove(chosen_worker)

        else:
            assignments[task.id] = []  # 任务未分配给任何工人

    return assignments


# 按质量/报酬的比值进行贪心分配
def greedy_quality_reward_ratio(workers, tasks, current_time, time_slot):
    time_slot = time_slot / 60
    # 根据任务类型分别计算剩余时间并排序
    tasks.sort(key=lambda x:
        (to_dt(x.start_time) + timedelta(hours=(len(x.assigned_workers) + 1) * x.interval) - current_time).seconds
        if x.task_type in ['TypeA', 'TypeB']
        else (to_dt(x.end_time) - current_time).seconds)

    assignments = {task.id: [] for task in tasks}  # 初始化字典以存储分配结果

    available_workers = workers.copy()
    assigned_worker_ids = set()  # 用于跟踪已分配的工人ID，另加

    for task in tasks:
        candidate_workers = []
        if task.task_type in ['TypeA', 'TypeB']:
            remaining_time = (to_dt(task.start_time) + timedelta(hours=(len(task.assigned_workers) + 1) * task.interval) - current_time).seconds   # 计算任务的剩余时间
        else:
            remaining_time = (to_dt(task.end_time) - current_time).seconds
        remaining_hours = remaining_time / 3600
        for worker in available_workers:
            if worker.id in assigned_worker_ids:  # 如果工人已经被分配，则跳过
                continue

            if not worker.is_available(current_time):  # 如果工人当前不可用，跳过此工人
                continue

            if task.task_type in ['TypeA', 'TypeB']:
                task_start = to_dt(task.start_time) + timedelta(hours=task.interval * len(task.assigned_workers))
                worker_start = calculate_worker_start_time(worker, current_time)
                start_task = worker_start if worker_start > task_start else task_start
            else:
                start_task = calculate_worker_start_time(worker, current_time)

            # 计算到达并完成任务需要的时间
            travel_time = calculate_travel_time(worker, task)
            # 检查 travel_time 是否超过2个分配时间间隔
            # if travel_time > 2 * time_slot:
            #     continue
            if travel_time > 2 * time_slot or (remaining_hours > 4 and travel_time > time_slot):
                continue

            sensing_time = calculate_pair_sensing_time(worker, task)
            total_time = travel_time + sensing_time
            # print(f"worker{worker.id}执行{task.task_type}的任务{task.id}，移动时间{travel_time}、执行时间{sensing_time}")

            # 检查工人完成任务的时间是否在工人的工作时间和任务截止时间之内
            expected_finish_time = start_task + timedelta(hours=total_time)
            if task.task_type not in ['TypeA', 'TypeB']:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= to_dt(task.end_time)):
                    sensing_quality = calculate_pair_sensing_quality(worker, task)
                    if sensing_quality >= task.quality_threshold:
                        reward = calculate_pair_sensing_bid(worker, task)
                        candidate_workers.append((worker, reward, sensing_quality, travel_time))
            else:
                if (expected_finish_time <= to_dt(worker.end_time) and
                        expected_finish_time <= (to_dt(task.start_time) + timedelta(
                            hours=(len(task.assigned_workers) + 1) * task.interval))):
                    sensing_quality = calculate_pair_sensing_quality(worker, task)
                    if sensing_quality >= task.quality_threshold:
                        reward = calculate_pair_sensing_bid(worker, task)
                        candidate_workers.append((worker, reward, sensing_quality, travel_time))

        # 按比值进行排序
        candidate_workers.sort(key=lambda x: x[2] / x[1], reverse=True)
        # print(f"{task.task_type}的{task.id}的候选工人数：{len(candidate_workers)}")

        # 只为任务分配一个合适的工人
        if candidate_workers:
            if task.task_type in ['TypeA', 'TypeB']:
                chosen_worker, _, _, travel_time = candidate_workers[0]
                assignments[task.id].append(chosen_worker.id)
                # print(f"选择的工人移动时间为：{travel_time}")
                assigned_worker_ids.add(chosen_worker.id)  # 记录已分配的工人
                available_workers.remove(chosen_worker)
            else:
                # 计算剩余可分配的工人数量
                remaining_workers_count = (task.num_area - len(task.assigned_workers))
                # 选择分配工人数量为剩余可分配数量和候选工人数量的较小值
                num_workers_to_assign = min(remaining_workers_count, len(candidate_workers))

                for i in range(num_workers_to_assign):
                    chosen_worker, _, _, travel_time = candidate_workers[i]
                    assignments[task.id].append(chosen_worker.id)
                    # print(f"选择的工人移动时间为：{travel_time}")
                    assigned_worker_ids.add(chosen_worker.id)  # 记录已分配的工人
                    available_workers.remove(chosen_worker)

        else:
            assignments[task.id] = []  # 任务未分配给任何工人

    return assignments
