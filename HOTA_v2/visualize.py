"""
可视化
"""
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_results(results):
    # 提取工人数量和对应的结果
    num_workers_list = list(results.keys())
    success_task_rates = [sum(data['success_task_rates']) / len(data['success_task_rates']) for data in
                          results.values()]
    utilizations = [sum(data['utilizations']) / len(data['utilizations']) for data in results.values()]
    total_distances = [sum(data['total_distances']) / len(data['total_distances']) for data in results.values()]

    # 绘制任务的平均成功分配率折线图
    plt.figure(figsize=(10, 5))
    plt.plot(num_workers_list, success_task_rates, marker='o', color='b', label='Success Task Rate')
    plt.xlabel('Number of Workers')
    plt.ylabel('Success Task Rate (%)')
    plt.title('Average Success Task Rate vs. Number of Workers')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制平均工人利用率折线图
    plt.figure(figsize=(10, 5))
    plt.plot(num_workers_list, utilizations, marker='o', color='r', label='Utilization')
    plt.xlabel('Number of Workers')
    plt.ylabel('Utilization (%)')
    plt.title('Average Utilization vs. Number of Workers')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制工人的总的移动距离的直方图
    plt.figure(figsize=(10, 6))
    plt.bar(num_workers_list, total_distances, alpha=0.5)
    plt.xlabel('Number of Workers')
    plt.ylabel('Total Distance (km)')
    plt.title('Total Distance by Number of Workers')
    plt.grid(True)
    plt.show()


def graph_visualize(g):
    # 创建一个绘图对象
    pos = nx.bipartite_layout(g, [n for n, d in g.nodes(data=True) if d['bipartite'] == 0])
    # 绘制节点和边
    nx.draw_networkx_nodes(g, pos, nodelist=[n for n, d in g.nodes(data=True) if d['bipartite'] == 0], node_color='b',
                           label='Workers')
    nx.draw_networkx_nodes(g, pos, nodelist=[n for n, d in g.nodes(data=True) if d['bipartite'] == 1], node_color='r',
                           label='Tasks')
    nx.draw_networkx_edges(g, pos)

    # 添加标签
    node_labels = {n: n.split('_')[1] for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=node_labels)

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


# 对二分图做可视化
def visualize_graph(g):
    # 创建一个图形和轴
    plt.figure(figsize=(12, 8))

    # 绘制图的布局
    pos = nx.bipartite_layout(g, nodes=[n for n, d in g.nodes(data=True) if d['bipartite'] == 0])

    # 绘制节点
    nx.draw_networkx_nodes(g, pos, nodelist=[n for n, d in g.nodes(data=True) if d['bipartite'] == 0],
                           node_color='lightblue', node_size=500, label='Workers')
    nx.draw_networkx_nodes(g, pos, nodelist=[n for n, d in g.nodes(data=True) if d['bipartite'] == 1],
                           node_color='lightgreen', node_size=500, label='Tasks')

    # 绘制边
    nx.draw_networkx_edges(g, pos)

    # 绘制标签
    nx.draw_networkx_labels(g, pos, labels={n: n for n in g.nodes()}, font_size=10, font_family='sans-serif')

    # 添加图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Workers'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Tasks')
    ]
    plt.legend(handles=legend_elements, loc='best')

    # 显示图形
    plt.title('Bipartite Graph of Workers and Tasks')
    plt.show()


def visualize_assignment(g, max_matching):
    # 创建一个绘图对象
    pos = nx.bipartite_layout(g, [n for n, d in g.nodes(data=True) if d['bipartite'] == 0])

    # 绘制节点和边
    nx.draw_networkx_nodes(g, pos, nodelist=[n for n, d in g.nodes(data=True) if d['bipartite'] == 0], node_color='b',
                           label='Workers')
    nx.draw_networkx_nodes(g, pos, nodelist=[n for n, d in g.nodes(data=True) if d['bipartite'] == 1], node_color='r',
                           label='Tasks')
    nx.draw_networkx_edges(g, pos)

    # 添加标签
    node_labels = {n: n.split('_')[1] for n in g.nodes()}
    nx.draw_networkx_labels(g, pos, labels=node_labels)

    # 绘制匹配边
    matching_edges = [(max_matching[worker], worker) for worker in max_matching]
    nx.draw_networkx_edges(g, pos, edgelist=matching_edges, edge_color='g', width=2)

    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()
