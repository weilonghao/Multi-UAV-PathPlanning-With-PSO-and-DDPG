import numpy as np
from scipy.spatial.distance import euclidean
import torch


class UAVEnv:
    """
    - 状态空间：包含无人机状态、目标点空间分布和全局进度
    - 动作空间：连续二维坐标建议（映射到最近可用目标）
    - 奖励函数：多目标综合奖励（生存概率+距离效率+探索激励）
    """

    def __init__(self, targets, survival_probs, uav_bases,
                 max_flight_distance=30000, num_uav=5):
        # 环境参数
        self.targets = np.array(targets, dtype=np.float32)
        self.survival_probs = np.array(survival_probs, dtype=np.float32)
        self.uav_bases = np.array(uav_bases, dtype=np.float32)
        self.num_targets = len(targets)
        self.num_uav = num_uav
        self.max_flight_distance = max_flight_distance

        # 状态空间维度计算
        self.state_dim = 3 + 3 * self.num_targets  # 基础状态 + 每个目标的3个特征

        # 初始化状态
        self.reset()

    def reset(self):
        """重置环境状态"""
        # UAV轨迹记录
        self.uav_distances = np.zeros(self.num_uav, dtype=np.float32)
        self.uav_paths = [[] for _ in range(self.num_uav)]

        # 当前决策状态
        self.current_uav_id = 0
        self.available_targets = list(range(self.num_targets))
        self.assigned_count = 0

        return self._get_state()

    def _get_state(self):

        state = np.zeros(self.state_dim, dtype=np.float32)

        # --- 基础状态 (3维) ---
        # 1. 当前无人机归一化ID [0,1]
        state[0] = self.current_uav_id / (self.num_uav - 1)

        # 2. 当前无人机航程使用率 [0,1]
        state[1] = self.uav_distances[self.current_uav_id] / self.max_flight_distance

        # 3. 剩余目标比例 [0,1]
        state[2] = len(self.available_targets) / self.num_targets

        # --- 目标点特征 (每个目标3维) ---
        base_x, base_y = self.uav_bases[self.current_uav_id]
        for i, target_id in enumerate(range(self.num_targets)):
            offset = 3 + i * 3

            if target_id in self.available_targets:
                # 相对坐标 (归一化)
                dx = (self.targets[target_id][0] - base_x) / 10000.0
                dy = (self.targets[target_id][1] - base_y) / 10000.0

                state[offset] = dx  # X相对坐标
                state[offset + 1] = dy  # Y相对坐标
                state[offset + 2] = self.survival_probs[target_id]  # 生存概率
            else:
                # 已分配目标用零填充
                state[offset:offset + 3] = 0.0

        return state

    def step(self, action):
        """
        执行动作（二维连续动作空间）
        action: [suggested_x, suggested_y] ∈ [0,1]^2

        """
        # 转换动作到地图坐标
        suggested_x = action[0] * 10000  # 映射到[0,10000]范围
        suggested_y = action[1] * 10000

        # --- 动作有效性处理 ---
        if not self.available_targets:
            return self._get_state(), 0.0, True, {"status": "all_assigned"}

        # 寻找最近可用目标
        selected, min_dist = -1, float('inf')
        for target_id in self.available_targets:
            dist = euclidean((suggested_x, suggested_y), self.targets[target_id])
            if dist < min_dist:
                min_dist = dist
                selected = target_id

        # --- 航程计算 ---
        current_base = self.uav_bases[self.current_uav_id]
        added_dist = 0.0

        if selected != -1:
            # 计算新增距离
            if not self.uav_paths[self.current_uav_id]:
                # 首个目标：基地->目标->基地
                added_dist = 2 * euclidean(current_base, self.targets[selected])
            else:
                last_target = self.uav_paths[self.current_uav_id][-1]
                old_dist = euclidean(self.targets[last_target], current_base)
                new_dist = (euclidean(self.targets[last_target], self.targets[selected]) +
                            euclidean(self.targets[selected], current_base))
                added_dist = new_dist - old_dist

            # 航程检查
            if (self.uav_distances[self.current_uav_id] + added_dist >
                    self.max_flight_distance):
                selected = -1  # 标记为无效选择

        reward = 0.0
        done = False

        if selected != -1:
            # 有效分配
            self.uav_paths[self.current_uav_id].append(selected)
            self.uav_distances[self.current_uav_id] += added_dist
            self.available_targets.remove(selected)
            self.assigned_count += 1

            #self._two_opt(self.current_uav_id)
            reward = self._calculate_reward(selected, added_dist)

            self.current_uav_id = (self.current_uav_id + 1) % self.num_uav
        else:
            # 无效动作惩罚
            reward = -2.0

        # 终止条件检查
        done = (len(self.available_targets) == 0) or (selected == -1)

        # 最终完成奖励
        if len(self.available_targets) == 0:

            reward += self._calculate_completion_bonus()

        return self._get_state(), reward, done, {"selected": selected}

    def _calculate_reward(self, target_id, added_dist):
        """优化后的奖励函数（强化聚类场景优势）"""
        # 动态参数计算
        progress = 1 - len(self.available_targets) / self.num_targets
        cluster_density = self.num_targets / (10000 ** 2)  # 假设环境为10km x 10km区域

        # 1. 强化生存价值奖励
        sp_reward = 1.6 * (self.survival_probs[target_id] ** 2)  # 平方项鼓励高价值目标

        # 2. 自适应距离惩罚
        base_dist_penalty = 0.6 * (added_dist / 1000)
        density_factor = 1 + 2 * cluster_density  # 高密度区域降低惩罚
        dist_penalty = base_dist_penalty / density_factor

        # 3. 集群探索激励
        cluster_bonus = 0.0
        if self.uav_paths[self.current_uav_id]:
            last_target = self.uav_paths[self.current_uav_id][-1]
            if euclidean(self.targets[last_target], self.targets[target_id]) < 1000:  # 1km内视为同一集群
                cluster_bonus = 1.2 * (1 - progress)  # 前期集群探索奖励更高

        # 4. 进度敏感激励（双阶段激励）
        if progress < 0.7:
            explore_bonus = 2.5 / (1 + np.exp(-8 * (progress - 0.3)))  # 早期快速提升
        else:
            explore_bonus = 3.0 * progress  # 后期线性增长

        # 5. 能效奖励（单位距离生存收益）
        efficiency = self.survival_probs[target_id] / (added_dist / 1000 + 1e-5)
        efficiency_bonus = 0.4 * np.log(efficiency + 1)  # 对数形式防止爆炸增长

        return sp_reward - dist_penalty + cluster_bonus + explore_bonus + efficiency_bonus

    def _two_opt(self, uav_id):

        path = self.uav_paths[uav_id]
        if len(path) < 3:
            return

        improved = True
        while improved:
            improved = False
            for i in range(1, len(path) - 1):
                for j in range(i + 1, len(path)):
                    if j - i == 1: continue
                    # 计算交换前后的路径长度差异
                    old_dist = (euclidean(self.targets[path[i - 1]], self.targets[path[i]]) +
                                euclidean(self.targets[path[j]], self.targets[path[(j + 1) % len(path)]]))
                    new_dist = (euclidean(self.targets[path[i - 1]], self.targets[path[j]]) +
                                euclidean(self.targets[path[i]], self.targets[path[(j + 1) % len(path)]]))
                    if new_dist < old_dist:
                        # 执行2-opt交换
                        path[i:j+1] = path[i:j+1][::-1]  # 反转路径段
                        improved = True
        # 更新优化后的路径
        self.uav_paths[uav_id] = path

    def _calculate_completion_bonus(self):
        if not any(len(path) > 0 for path in self.uav_paths):
            return 0.0

        # 平均路径效率计算
        total_efficiency = 0
        valid_paths = 0
        for uav_id in range(self.num_uav):
            path = self.uav_paths[uav_id]
            if len(path) > 1:
                base = self.uav_bases[uav_id]
                ideal_dist = 2 * euclidean(base, self.targets[path[0]])
                actual_dist = self.uav_distances[uav_id]
                total_efficiency += ideal_dist / actual_dist
                valid_paths += 1

        avg_efficiency = total_efficiency / valid_paths if valid_paths > 0 else 0

        # 综合奖励
        avg_survival = np.mean([np.prod([self.survival_probs[t] for t in path])
                                for path in self.uav_paths if path])
        return 1.5 * avg_efficiency * avg_survival

    def render(self):

        print(f"Current UAV: {self.current_uav_id}")
        print(f"Assigned targets: {self.assigned_count}/{self.num_targets}")
        print(f"Distances: {self.uav_distances}")
        print(f"Available targets: {len(self.available_targets)}")