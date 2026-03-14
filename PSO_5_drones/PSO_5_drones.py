import numpy as np
import random
from scipy.spatial.distance import euclidean
from statistics import variance
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'SimHei'
random.seed(42)
np.random.seed(42)
# ================== 配置参数 ==================
SHOW_CLUSTER_INFO = True  # 是否显示聚类过程信息
USE_CLUSTERED = True  # 是否使用聚类后目标点进行路径规划
CLUSTER_RADIUS = 500  # 聚类探测半径（米）
NUM_UAV = 5  # 无人机数量
MAX_FLIGHT_DISTANCE = 30000  # 无人机最大飞行距离（米）
SURVIVAL_RANGE = (0.7, 0.9)  # 生存概率生成范围
UAV_BASES = [(209, 391), (262, 338), (483, 401),(479, 284),(410, 368)]  # 无人机基地坐标
# =============================================

# ================== 数据准备 ==================
# 原始目标点数据（字典格式）
original_targets = {
    0: (6734, 1453), 1: (2233, 10), 2: (5530, 1424), 3: (401, 841), 4: (3082, 1644),
    5: (7608, 4458), 6: (7573, 3716), 7: (7265, 1268), 8: (6898, 1885), 9: (1112, 2049),
    10: (5468, 2606), 11: (5989, 2873), 12: (4706, 2674), 13: (4612, 2035), 14: (6347, 2683),
    15: (6107, 669), 16: (7611, 5184), 17: (7462, 3590), 18: (7732, 4723), 19: (5900, 3561),
    20: (4483, 3369), 21: (6101, 1110), 22: (5199, 2182), 23: (1633, 2809), 24: (4307, 2322),
    25: (675, 1006), 26: (7555, 4819), 27: (7541, 3981), 28: (3177, 756), 29: (7352, 4506),
    30: (7545, 2801), 31: (3245, 3305), 32: (6426, 3173), 33: (4608, 1198), 34: (23, 2216),
    35: (7248, 3779), 36: (7762, 4595), 37: (7392, 2244), 38: (3484, 2829), 39: (6271, 2135),
    40: (4985, 140), 41: (1916, 1569), 42: (7280, 4899), 43: (7509, 3239), 44: (10, 2676),
    45: (6807, 2993), 46: (5185, 3258), 47: (3023, 1942)
}

# 生成生存概率（原始目标点）
np.random.seed(42)
original_survival = np.round(np.random.uniform(*SURVIVAL_RANGE, len(original_targets)), 2).tolist()


## ================== 聚类算法 ==================
def perform_clustering(targets_dict, R=500):
    """执行聚类并返回聚类结果"""
    targets = list(targets_dict.values())
    n = len(targets)

    # 计算距离矩阵
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = euclidean(targets[i], targets[j])

    # 构建邻域关系
    neighborhoods = {i: [j for j in range(n) if j != i and dist_matrix[i][j] < R]
                     for i in range(n)}

    # 迭代聚类
    clusters = []
    assigned = set()
    while len(assigned) < n:
        candidates = [(i, len([j for j in neighborhoods[i] if j not in assigned]))
                      for i in range(n) if i not in assigned]
        best_candidate = max(candidates, key=lambda x: x[1])[0]
        cluster = [best_candidate] + [j for j in neighborhoods[best_candidate] if j not in assigned]
        clusters.append(cluster)
        assigned.update(cluster)

    # 计算聚类中心
    cluster_centers = []
    for cluster in clusters:
        x = np.mean([targets[i][0] for i in cluster])
        y = np.mean([targets[i][1] for i in cluster])
        cluster_centers.append((round(x, 2), round(y, 2)))

    # 计算聚类后的生存概率（取簇内平均值）
    clustered_survival = [np.mean([original_survival[i] for i in cluster]) for cluster in clusters]

    return cluster_centers, clusters, clustered_survival


# 执行聚类
clustered_targets, clusters, clustered_survival = perform_clustering(original_targets, CLUSTER_RADIUS)


# ================== 数据保存函数 ==================
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"数据已保存为 {filename}")


# 保存原始目标点数据
original_data = {
    "目标ID": list(original_targets.keys()),
    "X坐标": [t[0] for t in original_targets.values()],
    "Y坐标": [t[1] for t in original_targets.values()],
    "生存概率": original_survival
}
save_to_csv(original_data, "original_targets.csv")

# 保存聚类结果
clustered_data = {
    "聚类ID": list(range(len(clustered_targets))),
    "X坐标": [t[0] for t in clustered_targets],
    "Y坐标": [t[1] for t in clustered_targets],
    "平均生存概率": clustered_survival,
    "包含原始目标": [str([list(original_targets.keys())[i] for i in cluster]) for cluster in clusters]
}
save_to_csv(clustered_data, "clustered_targets.csv")


# ================== 路径规划算法 ==================
class PSOOptimizer:
    def __init__(self, targets, num_uav=3, max_L=30000, survival_probs=None, uav_bases=None):
        self.uav_bases = uav_bases if uav_bases else [(0, 0)] * num_uav
        self.targets = targets
        self.num_targets = len(targets)
        self.m = num_uav
        self.uav_speeds = [6.0, 7.0, 8.2,6.1,7.6][:num_uav]
        self.max_L = max_L
        self.survival_probs = survival_probs if survival_probs else [1.0] * self.num_targets

        # 按距离最近基地排序
        self.sorted_indices = sorted(range(self.num_targets),
                                     key=lambda i: min(euclidean(base, self.targets[i]) for base in self.uav_bases))

        # PSO参数
        self.num_particles = 200
        self.max_iter = 500
        self.w = 0.5
        self.c1 = 1.6
        self.c2 = 1.6

        # 优化权重
        self.alpha = 0.5  # 距离
        self.delta = 0.3  # 时间
        self.gamma = 0.1  # 负载
        self.beta = 0.1  # 存活概率

    class Particle:
        def __init__(self, num_targets):
            self.position = np.random.permutation(num_targets).tolist()
            self.velocity = np.random.rand(num_targets).tolist()
            self.best_position = self.position.copy()
            self.best_fitness = float('inf')
            self.fitness = float('inf')

    def optimize(self):
        particles = []
        for i in range(self.num_particles):
            if i == 0:
                p = self.Particle(self.num_targets)
                p.position = self.sorted_indices.copy()
                particles.append(p)
            else:
                particles.append(self.Particle(self.num_targets))

        global_best_position = None
        global_best_fitness = float('inf')
        fitness_history = []
        distance_history = []
        time_history = []
        load_history = []
        survival_history = []

        # 初始化粒子适应度
        for p in particles:
            p.fitness, components = self.fitness_function(p.position)
            if p.fitness < p.best_fitness:
                p.best_position = p.position.copy()
                p.best_fitness = p.fitness
            if p.fitness < global_best_fitness:
                global_best_position = p.position.copy()
                global_best_fitness = p.fitness

        fitness_history.append({
            "迭代次数": 0,
            "全局最优适应度": global_best_fitness,
            "平均适应度": np.mean([p.fitness for p in particles]),
            "最优粒子适应度": global_best_fitness
        })

        # 记录初始组件值
        _, init_components = self.fitness_function(global_best_position)
        distance_history.append(init_components['distance'])
        time_history.append(init_components['time'])
        load_history.append(init_components['load'])
        survival_history.append(init_components['survival'])

        # 迭代优化
        for iter in range(self.max_iter):
            for p in particles:
                # 更新速度
                new_velocity = [
                    self.w * v + self.c1 * random.random() * (pb - pos) + self.c2 * random.random() * (gb - pos)
                    for v, pos, pb, gb in zip(p.velocity, p.position, p.best_position, global_best_position)
                ]
                p.velocity = new_velocity

                # 更新位置
                new_pos_cont = [pos + v for pos, v in zip(p.position, p.velocity)]
                new_order = np.argsort(new_pos_cont).tolist()

                # 评估
                new_fitness, components = self.fitness_function(new_order)
                if new_fitness < p.best_fitness:
                    p.best_position = new_order.copy()
                    p.best_fitness = new_fitness
                if new_fitness < global_best_fitness:
                    global_best_position = new_order.copy()
                    global_best_fitness = new_fitness
                p.position = new_order
                p.fitness = new_fitness

            # 记录当前迭代信息
            current_avg = np.mean([p.fitness for p in particles])
            fitness_history.append({
                "迭代次数": iter + 1,
                "全局最优适应度": global_best_fitness,
                "平均适应度": current_avg,
                "最优粒子适应度": global_best_fitness
            })

            # 记录组件值
            _, components = self.fitness_function(global_best_position)
            distance_history.append(components['distance'])
            time_history.append(components['time'])
            load_history.append(components['load'])
            survival_history.append(components['survival'])

            print(f"Iter {iter + 1}: 最优适应度={global_best_fitness:.2f}, 平均适应度={current_avg:.2f}")

        # 获取最优解
        assignment = self.assign_targets(global_best_position)

        # 保存适应度组件数据
        component_data = {
            "迭代次数": list(range(len(distance_history))),
            "距离组件": distance_history,
            "时间组件": time_history,
            "负载组件": load_history,
            "生存概率组件": survival_history
        }
        save_to_csv(component_data, "fitness_components.csv")

        return assignment, fitness_history, (distance_history, time_history, load_history, survival_history)

    def assign_targets(self, order):
        uav_paths = [[] for _ in range(self.m)]
        uav_distances = [0.0] * self.m
        task_counts = [0] * self.m

        for target_idx in order:
            best_uav = -1
            min_cost = float('inf')

            for uav_id in range(self.m):
                current_base = self.uav_bases[uav_id]

                # 计算从基地出发，到目标并返回基地的飞行距离
                if not uav_paths[uav_id]:
                    # 首个目标：基地->目标->基地
                    new_dist = 2 * euclidean(current_base, self.targets[target_idx])
                else:
                    last = uav_paths[uav_id][-1]
                    add_dist = euclidean(self.targets[last], self.targets[target_idx])
                    new_dist = uav_distances[uav_id] + add_dist - euclidean(self.targets[last], current_base) + \
                               euclidean(self.targets[target_idx], current_base)

                # 检查航程限制
                if new_dist > self.max_L:
                    continue

                # 计算存活概率
                current_survival = np.prod([self.survival_probs[i] for i in uav_paths[uav_id] + [target_idx]])

                # 计算负载均衡代价
                balance = abs(task_counts[uav_id] - (self.num_targets / self.m)) ** 2

                # 综合成本计算
                cost = (new_dist * 0.6 +
                        balance * 0.2 +
                        (1 - current_survival) * 0.2)

                if cost < min_cost:
                    best_uav = uav_id
                    min_cost = cost

            # 处理无法分配的情况
            if best_uav == -1:
                min_exceed = float('inf')
                for uav_id in range(self.m):
                    current_base = self.uav_bases[uav_id]
                    if not uav_paths[uav_id]:
                        new_dist = 2 * euclidean(current_base, self.targets[target_idx])
                    else:
                        last = uav_paths[uav_id][-1]
                        new_dist = uav_distances[uav_id] + euclidean(self.targets[last],
                                                                     self.targets[target_idx]) + euclidean(
                            self.targets[target_idx], current_base) - euclidean(self.targets[last], current_base)
                    if new_dist < min_exceed:
                        best_uav = uav_id
                        min_exceed = new_dist

            # 执行分配
            current_base = self.uav_bases[best_uav]
            if not uav_paths[best_uav]:
                uav_distances[best_uav] = 2 * euclidean(current_base, self.targets[target_idx])
            else:
                last = uav_paths[best_uav][-1]
                uav_distances[best_uav] += euclidean(self.targets[last], self.targets[target_idx])
                uav_distances[best_uav] += euclidean(self.targets[target_idx], current_base)
                uav_distances[best_uav] -= euclidean(self.targets[last], current_base)

            uav_paths[best_uav].append(target_idx)
            task_counts[best_uav] += 1

        return uav_paths, uav_distances, task_counts

    def fitness_function(self, order):
        assignment = self.assign_targets(order)
        if not assignment:
            return float('inf'), {'distance': float('inf'), 'time': float('inf'),
                                  'load': float('inf'), 'survival': 0}

        uav_paths, uav_distances, task_counts = assignment

        # 计算总距离
        total_distance = sum(uav_distances)

        # 计算最大任务时间
        total_time = max([d / s for d, s in zip(uav_distances, self.uav_speeds) if d > 0])

        # 计算负载方差
        load_var = variance(task_counts) if len(task_counts) > 1 else 0

        # 计算平均存活概率
        survival_probs = []
        for path in uav_paths:
            if path:
                survival_probs.append(np.prod([self.survival_probs[i] for i in path]))
        avg_survival = np.mean(survival_probs) if survival_probs else 0

        # 归一化处理
        norm_dist = total_distance / (self.max_L * self.m)
        norm_time = total_time / 3600
        norm_load = load_var / self.num_targets
        norm_survival = 1 - avg_survival

        fitness = (self.alpha * norm_dist +
                   self.delta * norm_time +
                   self.gamma * norm_load +
                   self.beta * norm_survival)

        components = {
            'distance': norm_dist,
            'time': norm_time,
            'load': norm_load,
            'survival': norm_survival
        }

        return fitness, components

    def plot_results(self, paths, survival_probs, title):
        plt.figure(figsize=(12, 8))
        # 绘制目标点
        sc = plt.scatter([t[0] for t in self.targets], [t[1] for t in self.targets],
                         c=survival_probs, cmap='coolwarm', marker='o',
                         vmin=0.6, vmax=1.0, label='目标点')

        # 绘制基地
        for idx, base in enumerate(self.uav_bases):
            plt.scatter(base[0], base[1], c='black', marker='s', s=100, label=f'基地{idx + 1}')

        colors = ['red', 'green', 'blue', 'purple', 'cyan']
        for uav_id, path in enumerate(paths):
            if not path:
                continue
            # 绘制路径
            base = self.uav_bases[uav_id]
            x = [base[0]] + [self.targets[i][0] for i in path] + [base[0]]
            y = [base[1]] + [self.targets[i][1] for i in path] + [base[1]]
            plt.plot(x, y, marker='o', color=colors[uav_id],
                     label=f'UAV{uav_id + 1}')

        plt.title(title)
        plt.legend()
        plt.show()


# ================== 执行优化 ==================
def run_optimization(targets, survival_probs, is_clustered=False):
    print("\n" + "=" * 40)
    print(f"正在执行{'聚类后' if is_clustered else '原始'}目标点优化...")

    optimizer = PSOOptimizer(targets, NUM_UAV, MAX_FLIGHT_DISTANCE,
                             survival_probs, UAV_BASES)
    assignment, fitness_curve, components = optimizer.optimize()

    # 保存适应度数据
    save_to_csv(fitness_curve, f"{'clustered' if is_clustered else 'original'}_fitness.csv")
    save_to_csv(components, f"{'clustered' if is_clustered else 'original'}_components.csv")
    if assignment:
        paths, distances, counts = assignment
        total_distance = sum(distances)
        total_time = max([d / s for d, s in zip(distances, optimizer.uav_speeds)])

        # 保存路径数据
        path_data = []
        for uav_id in range(len(paths)):
            path = paths[uav_id]
            for order, target_idx in enumerate(path):
                path_data.append({
                    "无人机ID": uav_id + 1,
                    "路径顺序": order + 1,
                    "目标X": targets[target_idx][0],
                    "目标Y": targets[target_idx][1],
                    "生存概率": survival_probs[target_idx],
                    "飞行距离": distances[uav_id],
                    "任务时间": distances[uav_id] / optimizer.uav_speeds[uav_id]
                })
        save_to_csv(path_data, f"{'clustered' if is_clustered else 'original'}_paths.csv")

        # 绘制路径图
        optimizer.plot_results(paths, survival_probs,
                               f"{'聚类后' if is_clustered else '原始'}目标点路径规划")

        # 绘制适应度曲线
        plt.figure(figsize=(12, 6))
        plt.plot([f['全局最优适应度'] for f in fitness_curve], label='全局最优适应度')
        plt.title("适应度收敛曲线")
        plt.xlabel("迭代次数")
        plt.ylabel("适应度值")
        plt.legend()
        plt.show()

        # 绘制组件曲线
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(components[0], label='距离')
        plt.title("距离变化")
        plt.xlabel("迭代次数")
        plt.ylabel("距离")

        plt.subplot(2, 2, 2)
        plt.plot(components[1], label='时间')
        plt.title("时间变化")
        plt.xlabel("迭代次数")
        plt.ylabel("时间")

        plt.subplot(2, 2, 3)
        plt.plot(components[2], label='负载')
        plt.title("负载变化")
        plt.xlabel("迭代次数")
        plt.ylabel("负载方差")

        plt.subplot(2, 2, 4)
        plt.plot(components[3], label='生存概率')
        plt.title("生存概率变化")
        plt.xlabel("迭代次数")
        plt.ylabel("生存概率")

        plt.tight_layout()
        plt.show()

        return fitness_curve, components
    return None, None


# ================== 主流程 ==================
if __name__ == "__main__":
    all_fitness = []
    all_components = []

    original_points = list(original_targets.values())
    original_curve, original_comps = run_optimization(original_points, original_survival, is_clustered=False)
    if original_curve:
        all_fitness.append(("原始", original_curve))
        all_components.append(("原始", original_comps))

    if USE_CLUSTERED:
        clustered_curve, clustered_comps = run_optimization(clustered_targets, clustered_survival, is_clustered=True)
        if clustered_curve:
            all_fitness.append(("聚类后", clustered_curve))
            all_components.append(("聚类后", clustered_comps))

    # 绘制对比曲线
    plt.figure(figsize=(12, 6))
    for label, curve in all_fitness:
        plt.plot([f['全局最优适应度'] for f in curve], label=f'{label}最优适应度')
    plt.title("优化过程收敛曲线对比")
    plt.xlabel("迭代次数")
    plt.ylabel("适应度值")
    plt.legend()
    plt.show()

    # 绘制组件对比曲线
    if len(all_components) > 1:
        component_names = ['距离', '时间', '负载', '生存概率']
        plt.figure(figsize=(15, 10))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            for label, comps in all_components:
                plt.plot(comps[i], label=f'{label}{component_names[i]}')
            plt.title(f"{component_names[i]}")
            plt.xlabel("迭代次数")
            plt.ylabel("归一化值")
            plt.legend()
        plt.tight_layout()
        plt.show()