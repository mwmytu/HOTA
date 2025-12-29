def fitness(individual, tasks, workers):
    fitness_score = 0

    for task_id, assigned_workers in individual.items():
        task = tasks[task_id]
        total_cost = 0

        # Check if the number of assigned workers meets the requirement
        if len(assigned_workers) < task.required_workers:
            return -float('inf')  # Return a very low score if the requirement is not met

        # Check if all workers meet the task's budget and time requirements
        for worker_id in assigned_workers:
            worker = workers[worker_id]
            if worker.end_time < task.deadline or task.budget < worker.quote:
                return -float('inf')  # Return a very low score if conditions are not met

            total_cost += worker.quote

        # Check if total cost is within the budget
        if total_cost > task.budget:
            return -float('inf')  # Return a very low score if budget is exceeded

        fitness_score += calculate_task_score(task, assigned_workers)  # Define how to score the task

    return fitness_score


def create_individual(workers, tasks, current_time):
    possible_pairs = [(w, t) for w in workers for t in tasks]
    # 随机打乱配对列表
    random.shuffle(possible_pairs)

    # 初始化解
    individual = {task.id: [] for task in tasks}

    # 遍历打乱后的配对列表，尝试分配
    for worker, task in possible_pairs:
        # 检查工人是否已经分配
        if not worker.is_available(current_time):
            continue

        # 处理任务类型分配需求
        if task.task_type in ['TypeA', 'TypeB']:
            required_workers_count = 1
        else:
            required_workers_count = len(task.location) - len(task.assigned_workers)

        # 检查任务是否已经满足所需的工人数
        if len(individual[task.id]) >= required_workers_count:
            continue

        # 计算工人的开始时间
        if task.task_type in ['TypeA', 'TypeB']:
            task_start = to_dt(task.start_time) + timedelta(hours=len(task.assigned_workers) * task.interval)
            worker_start = calculate_worker_start_time(worker, current_time)
            start_task = max(worker_start, task_start)
        else:
            start_task = calculate_worker_start_time(worker, current_time)

        # 计算到达并完成任务需要的时间
        travel_time = calculate_travel_time(worker, task)
        sensing_time = task.sensing_time / 60
        total_time = travel_time + sensing_time

        # 如果工人完成任务的时间满足时间、预算限制约束，加入该任务的候选工人列表
        expected_finish_time = start_task + timedelta(hours=total_time)
        if task.task_type not in ['TypeA', 'TypeB']:
            if (expected_finish_time <= worker.end_time and
                    expected_finish_time <= task.end_time):
                bid = calculate_bid(worker, task)
                if bid <= calculate_avg_budget(task):
                    individual[task.id].append(worker.id)

        else:
            if (expected_finish_time <= worker.end_time and
                    expected_finish_time <= (task.start_time + timedelta(hours=len(task.assigned_workers) + 1))):
                bid = calculate_bid(worker, task)
                if bid <= calculate_avg_budget(task):
                    individual[task.id].append(worker.id)

    return individual


