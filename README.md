HOTA_v1 performs simulations on synthetic dataset. The main function executes different algorithms, and the task distribution —uniform, concentrated, and mixed—can be selected and modified in main_task_fixed() and main_worker_fixed(). HOTA_v2 filters files from the t_drive dataset whose trajectory points meet a certain threshold, assigning them as user trajectories. It then determines users' real-time locations based on the simulated time for task assignment. Similarly, the main function runs different algorithms, and the task distribution are adjusted within main_task_fixed() and main_worker_fixed() as uniform, concentrated, or mixed.

All configurations are modified within the main_task_fixed() (for fixed tasks) and main_worker_fixed() (for fixed workers) functions:

· algorithms_to_run: Edit the list to select the algorithms to run, e.g., ['GR', 'MFTA', 'GGA-I', 'BPGA'].
· num_experiments: Set the number of repeated trials per parameter group (for averaging results).
· num_workers_list / num_tasks_list: Define the range of variations for the independent variables.
· ga_params: Adjust parameters for the genetic algorithm (e.g., number of iterations, population size, etc.).

After running the program, results are automatically saved to the corresponding directories:

· *_detailed.csv: Contains detailed data from each experiment (for debugging or variance analysis).
· *_avg.csv: Contains averaged data from multiple experiments.

Link to the paper:
https://www.sciencedirect.com/science/article/pii/S0140366425002269
