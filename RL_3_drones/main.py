import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import pandas as pd


from data_utils import original_targets, original_survival
from cluster_utils import perform_clustering
from env import UAVEnv
from rl_optimizer import DDPGAgent
from config import (
    USE_CLUSTERED, CLUSTER_RADIUS,
    NUM_UAV, MAX_FLIGHT_DISTANCE, UAV_BASES,
    DDPG_MAX_EPISODES, DDPG_MAX_STEPS_PER_EPISODE, DDPG_BATCH_SIZE, RANDOM_SEED
)

# 创建结果目录
os.makedirs("result", exist_ok=True)
plt.rcParams['axes.unicode_minus'] = False # 显示负号

def save_to_csv(data, filename):
    """
    保存数据到 CSV 文件（文件保存在 result 目录下）
    """
    df = pd.DataFrame(data)
    df.to_csv(f"result/{filename}", index=False, encoding='utf-8-sig')
    print(f"数据已保存为 {filename}")


def set_global_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_clustering():
    """执行目标点聚类并保存结果"""
    # 保存原始目标点数据
    original_data = {
        "目标ID": list(original_targets.keys()),
        "X坐标": [t[0] for t in original_targets.values()],
        "Y坐标": [t[1] for t in original_targets.values()],
        "生存概率": original_survival
    }
    save_to_csv(original_data, "original_targets.csv")

    # 执行聚类
    cluster_centers, clusters, clustered_survival = perform_clustering(
        original_targets, R=CLUSTER_RADIUS, original_survival=original_survival
    )

    # 保存聚类结果
    clustered_data = {
        "聚类ID": list(range(len(cluster_centers))),
        "X坐标": [pt[0] for pt in cluster_centers],
        "Y坐标": [pt[1] for pt in cluster_centers],
        "平均生存概率": clustered_survival,
        "包含原始目标": [str([list(original_targets.keys())[i] for i in cluster]) for cluster in clusters]
    }
    save_to_csv(clustered_data, "clustered_targets.csv")

    return cluster_centers, clustered_survival


def train_agent(env, agent, max_episodes, max_steps, tag=""):
    """训练循环与性能监控"""
    best_reward = -np.inf
    reward_history = []
    episode_lengths = []
    success_rates = []
    loss_history = {'critic': [], 'actor': []}

    progress_bar = tqdm(range(max_episodes), desc=f"Training {tag}")

    for ep in progress_bar:
        state = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done and ep_steps < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 异步更新（每10步更新一次）
            if ep_steps % 10 == 0 and len(agent.replay_buffer) > DDPG_BATCH_SIZE * 10:
                critic_loss, actor_loss = agent.update()
                if critic_loss is not None:
                    loss_history['critic'].append(critic_loss)
                    loss_history['actor'].append(actor_loss)

            state = next_state
            ep_reward += reward
            ep_steps += 1

        # 记录指标
        success = int(len(env.available_targets) == 0)
        reward_history.append(ep_reward)
        episode_lengths.append(ep_steps)
        success_rates.append(success)

        progress_bar.set_postfix({
            "Reward": f"{ep_reward:.1f}",
            "Success": f"{np.mean(success_rates[-100:]):.2f}",
            "Noise": f"{agent.noise_scale:.3f}"
        })

        # 保存最佳模型
        if ep_reward > best_reward and success:
            best_reward = ep_reward
            agent.save(f"result/best_model_{tag}.pth")

    return reward_history, episode_lengths, success_rates, loss_history


def evaluate_agent(env, agent, num_episodes=10):
    """评估训练好的智能体"""
    success_count = 0
    survival_rates = []
    travel_distances = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, add_noise=False)
            state, _, done, _ = env.step(action)

        # 计算评估指标
        success = int(len(env.available_targets) == 0)
        success_count += success

        if success:
            survivals = [
                np.prod([env.survival_probs[t] for t in path])
                for path in env.uav_paths if path
            ]
            survival_rates.append(np.mean(survivals))
            travel_distances.extend(env.uav_distances)

    success_rate = success_count / num_episodes
    avg_survival = np.mean(survival_rates) if survival_rates else 0.0
    avg_distance = np.mean(travel_distances) if travel_distances else 0.0

    return success_rate, avg_survival, avg_distance


def visualize_results(targets, survival_probs, paths, distances, title):
    """可视化最终路径规划结果"""
    plt.figure(figsize=(12, 8))

    sc = plt.scatter(
        [t[0] for t in targets],
        [t[1] for t in targets],
        c=survival_probs,
        cmap='coolwarm',
        vmin=0.6,
        vmax=1.0,
        marker='o',
        label='目标点'
    )
    # 绘制无人机基地
    for idx, base in enumerate(UAV_BASES):
        plt.scatter(base[0], base[1], c='black', marker='s', s=100, label=f'基地{idx + 1}')

    # 绘制路径
    colors = ['red', 'green', 'blue', 'purple', 'cyan']
    for uav_id, (path, dist) in enumerate(zip(paths, distances)):
        if not path:
            continue

        x = [UAV_BASES[uav_id][0]] + [targets[i][0] for i in path] + [UAV_BASES[uav_id][0]]
        y = [UAV_BASES[uav_id][1]] + [targets[i][1] for i in path] + [UAV_BASES[uav_id][1]]

        plt.plot(x, y,
                 marker='o',
                 color=colors[uav_id % len(colors)],
                 linewidth=2,
                 markersize=6,
                 label=f'UAV{uav_id + 1} (距离: {dist / 1000:.1f}km)')

    plt.title(f"{title}\n总飞行距离: {sum(distances) / 1000:.1f}km")
    plt.xlabel("X坐标 (米)")
    plt.ylabel("Y坐标 (米)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"result/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(rewards, success_rates, losses, tag):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))

    # 奖励曲线
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title("回合奖励")
    plt.xlabel("训练轮次")

    # 损失曲线
    plt.subplot(1, 3, 3)
    plt.plot(losses['critic'], label='Critic Loss')
    plt.plot(losses['actor'], label='Actor Loss')
    plt.title("网络损失")
    plt.xlabel("更新次数")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"result/training_curves_{tag}.png", dpi=300)
    plt.show()


def extract_uav_data(env, targets, survival_probs):

    mission_times = getattr(env, "uav_times", [0.0] * len(env.uav_paths))
    rows = []
    uav_speed = [6.0,7.0,8.2,6.1,7.6]
    for uav_idx, path in enumerate(env.uav_paths):
        flight_distance = env.uav_distances[uav_idx] if uav_idx < len(env.uav_distances) else 0.0
        mission_time = flight_distance/uav_speed[uav_idx]
        for seq, target_index in enumerate(path, start=1):
            # 从 targets 中提取目标坐标，注意 targets 为列表，每个元素为 (X, Y)
            target = targets[target_index]
            target_x, target_y = target[0], target[1]
            # 获取对应目标的生存概率
            surv_prob = survival_probs[target_index]
            rows.append({
                "无人机ID": uav_idx + 1,
                "路径顺序": seq,
                "目标X": target_x,
                "目标Y": target_y,
                "生存概率": surv_prob,
                "飞行距离": flight_distance,
                "任务时间": mission_time
            })
    return rows


def run_optimization(targets, survival_probs, is_clustered=False):
    tag = "clustered" if is_clustered else "original"

    # 初始化环境和智能体
    env = UAVEnv(
        targets=targets,
        survival_probs=survival_probs,
        uav_bases=UAV_BASES,
        max_flight_distance=MAX_FLIGHT_DISTANCE,
        num_uav=NUM_UAV
    )
    agent = DDPGAgent(state_dim=env.state_dim)

    print(f"\n=== 开始 {tag.upper()} 目标点优化 ===")
    print(f"目标点数量: {len(targets)}")
    print(f"状态维度: {env.state_dim}")
    print(f"最大训练轮次: {DDPG_MAX_EPISODES}")

    # 训练阶段
    train_rewards, ep_lengths, success_rates, loss_history = train_agent(
        env, agent, DDPG_MAX_EPISODES, DDPG_MAX_STEPS_PER_EPISODE, tag
    )

    # 绘制训练曲线
    plot_training_curves(train_rewards, success_rates, loss_history, tag)

    # 评估阶段
    test_env = UAVEnv(
        targets=targets,
        survival_probs=survival_probs,
        uav_bases=UAV_BASES,
        max_flight_distance=MAX_FLIGHT_DISTANCE,
        num_uav=NUM_UAV
    )
    success_rate, avg_survival, avg_distance = evaluate_agent(test_env, agent)

    print(f"\n评估结果 ({tag}):")
    print(f"成功率: {success_rate * 100:.1f}%")
    print(f"平均飞行距离: {avg_distance / 1000:.1f}km")

    # 保存评估结果到 CSV 文件
    evaluation_results = {
        "优化类型": ["聚类" if is_clustered else "原始"],
        "成功率(%)": [success_rate * 100],
        "平均生存概率": [avg_survival],
        "平均飞行距离(km)": [avg_distance / 1000]
    }
    save_to_csv(evaluation_results, f"evaluation_results_{tag}.csv")

    if success_rate > 0:
        agent.load(f"result/best_model_{tag}.pth")
        state = test_env.reset()
        done = False
        while not done:
            action = agent.select_action(state, add_noise=False)
            state, _, done, _ = test_env.step(action)

        visualize_results(
            targets=targets,
            survival_probs=survival_probs,
            paths=test_env.uav_paths,
            distances=test_env.uav_distances,
            title=f"DDPG路径规划-{tag}"
        )

        # 从 test_env 中提取无人机数据并保存
        uav_data = extract_uav_data(test_env, targets, survival_probs)
        save_to_csv(uav_data, f"uav_results_{tag}.csv")

    return train_rewards


def plot_curves(rewards_original, rewards_clustered):
    """绘制原始与聚类目标点优化的训练曲线"""
    plt.figure(figsize=(10, 6))

    # 奖励曲线
    plt.plot(rewards_original, label="原始目标点优化", color='blue')
    plt.plot(rewards_clustered, label="聚类目标点优化", color='red')

    plt.title("奖励曲线对比")
    plt.xlabel("训练轮次")
    plt.ylabel("奖励")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"result/training_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # 1. 数据准备与聚类
    cluster_centers, clustered_survival = setup_clustering()
    set_global_seeds(RANDOM_SEED)

    # 2. 原始目标点优化
    reward_original = run_optimization(
        targets=list(original_targets.values()),
        survival_probs=original_survival,
        is_clustered=False
    )

    # 3. 聚类目标点优化
    if USE_CLUSTERED:
        reward_cluster = run_optimization(
            targets=cluster_centers,
            survival_probs=clustered_survival,
            is_clustered=True
        )
    save_to_csv(reward_original,"reward_original.csv")
    save_to_csv(reward_cluster,"reward_cluster.csv")
    plot_curves(reward_original,reward_cluster)

