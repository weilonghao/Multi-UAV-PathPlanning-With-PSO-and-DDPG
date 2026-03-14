#config.py
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'

# ================== 配置参数 ==================
USE_CLUSTERED = True           # 是否使用聚类后的目标点
CLUSTER_RADIUS = 500           # 聚类探测半径（米）
NUM_UAV = 5                    # 无人机数量
MAX_FLIGHT_DISTANCE = 30000    # 无人机最大飞行距离（米）
SURVIVAL_RANGE = (0.7, 0.9)    # 生存概率生成范围

UAV_BASES = [                  # 无人机基地坐标
    (209, 391),
    (262, 338),
    (483, 401),
    (479, 284),
    (410, 368)
]

# ========== DDPG 超参数 ==========
DDPG_MAX_EPISODES = 1000
DDPG_MAX_STEPS_PER_EPISODE = 500
DDPG_ACTOR_LR = 1e-3
DDPG_CRITIC_LR = 1e-3
DDPG_GAMMA = 0.95
DDPG_TAU = 0.01
DDPG_BATCH_SIZE = 64
DDPG_BUFFER_SIZE = 100000

RANDOM_SEED = 10