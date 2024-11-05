import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results():
    # 보상 데이터 로드
    rewards = pd.read_csv(f'results/rewards.csv')
    episodes = rewards[:, 0]
    reward_values = rewards[:, 1]

    # 메트릭 데이터 로드
    metrics = pd.read_csv(f'results/metrics.csv')
    steps = metrics[:, 0]
    critic1_losses = metrics[:, 1]
    critic2_losses = metrics[:, 2]
    actor_losses = metrics[:, 3]
    log_pis = metrics[:, 4]
    q_values = metrics[:, 5]

    # 그래프 생성
    plt.figure(figsize=(15, 10))

    # 보상 그래프
    plt.subplot(2, 3, 1)
    plt.plot(episodes, reward_values)
    plt.title('Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Critic Losses
    plt.subplot(2, 3, 2)
    plt.plot(steps, critic1_losses, label='Critic1')
    plt.plot(steps, critic2_losses, label='Critic2')
    plt.title('Critic Losses')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()

    # Actor Loss
    plt.subplot(2, 3, 3)
    plt.plot(steps, actor_losses)
    plt.title('Actor Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')

    # Log Pi
    plt.subplot(2, 3, 4)
    plt.plot(steps, log_pis)
    plt.title('Log Pi')
    plt.xlabel('Step')
    plt.ylabel('Value')

    # Q Values
    plt.subplot(2, 3, 5)
    plt.plot(steps, q_values)
    plt.title('Q Values')
    plt.xlabel('Step')
    plt.ylabel('Value')

    plt.tight_layout()
    plt.savefig(f'results/training_plots.png')
    plt.close()

if __name__ == "__main__":
    plot_training_results()