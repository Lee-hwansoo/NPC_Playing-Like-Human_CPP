import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results(result_dir):
   # CSV 파일 로드
   rewards_df = pd.read_csv(os.path.join(result_dir, 'rewards.csv'))
   metrics_df = pd.read_csv(os.path.join(result_dir, 'metrics.csv'))

   fig = plt.figure(figsize=(15, 10))

   # 1. 보상 그래프
   plt.subplot(2, 3, 1)
   plt.plot(rewards_df['episode'], rewards_df['reward'], color='blue')
   plt.title('Episode Rewards')
   plt.xlabel('Episode')
   plt.ylabel('Reward')

   # Critic Loss 그래프들의 공통 y축 범위 계산
   y_min = min(metrics_df['critic_loss1'].min(), metrics_df['critic_loss2'].min())
   y_max = max(metrics_df['critic_loss1'].max(), metrics_df['critic_loss2'].max())

   # 2. Critic1 Loss 그래프
   plt.subplot(2, 3, 2)
   plt.plot(metrics_df['step'], metrics_df['critic_loss1'], color='red')
   plt.ylim(y_min, y_max)
   plt.title('Critic1 Loss')
   plt.xlabel('Step')
   plt.ylabel('Loss')

   # 3. Critic2 Loss 그래프
   plt.subplot(2, 3, 3)
   plt.plot(metrics_df['step'], metrics_df['critic_loss2'], color='green')
   plt.ylim(y_min, y_max)
   plt.title('Critic2 Loss')
   plt.xlabel('Step')
   plt.ylabel('Loss')

   # 4. Actor Loss 그래프
   plt.subplot(2, 3, 4)
   plt.plot(metrics_df['step'], metrics_df['actor_loss'], color='blue')
   plt.title('Actor Loss')
   plt.xlabel('Step')
   plt.ylabel('Loss')

   # 5. Log Pi 그래프
   plt.subplot(2, 3, 5)
   plt.plot(metrics_df['step'], metrics_df['log_pi'], color='red')
   plt.title('Log Pi')
   plt.xlabel('Step')
   plt.ylabel('Value')

   # 6. Q Value 그래프
   plt.subplot(2, 3, 6)
   plt.plot(metrics_df['step'], metrics_df['q_value'], color='purple')
   plt.title('Q Values')
   plt.xlabel('Step')
   plt.ylabel('Value')

   plt.tight_layout()

   # 그래프 표시
   plt.show()

if __name__ == "__main__":
   result_dir = "out/results/environment"
   plot_training_results(result_dir)