import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results(result_dir):
   # CSV 파일 로드
   rewards_df = pd.read_csv(os.path.join(result_dir, 'train_rewards.csv'))
   metrics_df = pd.read_csv(os.path.join(result_dir, 'train_metrics.csv'))

   fig = plt.figure(figsize=(20, 10))

   # 1. 보상 그래프
   plt.subplot(2, 4, 1)
   plt.plot(rewards_df['episode'], rewards_df['reward'], color='blue')
   plt.title('Episode Rewards')
   plt.xlabel('Episode')
   plt.ylabel('Reward')

   # Critic Loss 그래프들의 공통 y축 범위 계산
   y_min = min(metrics_df['critic_loss1'].min(), metrics_df['critic_loss2'].min())
   y_max = max(metrics_df['critic_loss1'].max(), metrics_df['critic_loss2'].max())

   # 2. Critic1 Loss 그래프
   plt.subplot(2, 4, 2)
   plt.plot(metrics_df['step'], metrics_df['critic_loss1'], color='red')
   plt.ylim(y_min, y_max)
   plt.title('Critic1 Loss')
   plt.xlabel('Step')
   plt.ylabel('Loss')

   # 3. Critic2 Loss 그래프
   plt.subplot(2, 4, 3)
   plt.plot(metrics_df['step'], metrics_df['critic_loss2'], color='green')
   plt.ylim(y_min, y_max)
   plt.title('Critic2 Loss')
   plt.xlabel('Step')
   plt.ylabel('Loss')

   # 4. Combined Critics Loss
   plt.subplot(2, 4, 4)
   plt.plot(metrics_df['step'], metrics_df['critic_loss1'], color='red', alpha=1, label='Critic1')
   plt.plot(metrics_df['step'], metrics_df['critic_loss2'], color='green', alpha=0.5, label='Critic2')
   plt.ylim(y_min, y_max)
   plt.title('Combined Critics Loss')
   plt.xlabel('Step')
   plt.ylabel('Loss')
   plt.legend()

   # 5. Actor Loss 그래프
   plt.subplot(2, 4, 5)
   plt.plot(metrics_df['step'], metrics_df['actor_loss'], color='blue')
   plt.title('Actor Loss')
   plt.xlabel('Step')
   plt.ylabel('Loss')

   # 6. Log Pi 그래프
   plt.subplot(2, 4, 6)
   plt.plot(metrics_df['step'], metrics_df['log_pi'], color='purple')
   plt.title('Log Pi')
   plt.xlabel('Step')
   plt.ylabel('Value')

   # 7. Q Value 그래프
   plt.subplot(2, 4, 7)
   plt.plot(metrics_df['step'], metrics_df['q_value'], color='purple')
   plt.title('Q Values')
   plt.xlabel('Step')
   plt.ylabel('Value')

   plt.tight_layout()

   # 그래프 표시
   plt.show()

if __name__ == "__main__":
   print("현재 디렉토리 절대 경로:", os.getcwd())
   # result_dir = os.getcwd() + "/../../out/results/environment"
   result_dir = os.getcwd() + "/results/environment"
   plot_training_results(result_dir)