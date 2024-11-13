import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_training_results(result_dir):
    # CSV 파일 로드
    rewards_df = pd.read_csv(os.path.join(result_dir, 'train_results.csv'))
    metrics_df = pd.read_csv(os.path.join(result_dir, 'train_metrics.csv'))

    # Seaborn 스타일 설정
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # 서브플롯 레이아웃 개선
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 5, figure=fig)

    # 1. 보상 그래프 (더 큰 크기로)
    ax1 = fig.add_subplot(gs[0, :2])  # 첫 번째 행의 2/5를 차지
    sns.lineplot(data=rewards_df, x='episode', y='reward', color='blue', ax=ax1)
    ax1.set_title('Episode Rewards', size=12, pad=10)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    # 이동평균선 추가
    window_size = 50
    rewards_df['rolling_mean'] = rewards_df['reward'].rolling(window=window_size).mean()
    sns.lineplot(data=rewards_df, x='episode', y='rolling_mean', color='red', ax=ax1, label=f'Moving Average (window={window_size})')
    ax1.legend()

    # 1-2. 도착 성공률 그래프 추가
    ax1_2 = fig.add_subplot(gs[0, 2])  # 첫 번째 행의 1/5를 차지
    # 1-2. 1. 실제 성공/실패를 점으로 표시
    sns.scatterplot(data=rewards_df, x='episode', y=rewards_df['arrived']*100, color='lightgreen', alpha=0.3, ax=ax1_2)

    # 1-2. 2. 이동 평균으로 성공률 계산 (window_size는 더 작게 조정 가능)
    window_size_arrival = 10  # 도착률용 윈도우 사이즈는 별도로 설정
    rewards_df['arrival_rate'] = rewards_df['arrived'].rolling(window=window_size_arrival).mean() * 100

    # 1-2. 3. 이동 평균 선 추가
    sns.lineplot(data=rewards_df, x='episode', y='arrival_rate',color='green', ax=ax1_2, label=f'Moving Average (window={window_size_arrival})')

    ax1_2.set_title('Arrival Success Rate (%)', size=12, pad=10)
    ax1_2.set_xlabel('Episode')
    ax1_2.set_ylabel('Success Rate (%)')
    ax1_2.set_ylim(-2, 120)

    # 1-2. 범례 추가
    ax1_2.legend(loc='upper left')

    # Critic Loss 그래프들의 공통 y축 범위 계산
    y_min = min(metrics_df['critic_loss1'].min(), metrics_df['critic_loss2'].min())
    y_max = max(metrics_df['critic_loss1'].max(), metrics_df['critic_loss2'].max())

    # 2. Combined Critics Loss (더 큰 크기로)
    ax2 = fig.add_subplot(gs[0, 3:])  # 첫 번째 행의 나머지 절반
    sns.lineplot(data=metrics_df, x='step', y='critic_loss1', color='red',
                label='Critic1', ax=ax2, alpha=0.7)
    sns.lineplot(data=metrics_df, x='step', y='critic_loss2', color='green',
                label='Critic2', ax=ax2, alpha=0.7)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title('Combined Critics Loss', size=12, pad=10)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # 3. Actor Loss
    ax3 = fig.add_subplot(gs[1, 0])
    sns.lineplot(data=metrics_df, x='step', y='actor_loss', color='blue', ax=ax3)
    ax3.set_title('Actor Loss', size=12, pad=10)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Loss')

    # 4. Log Pi
    ax4 = fig.add_subplot(gs[1, 1])
    sns.lineplot(data=metrics_df, x='step', y='log_pi', color='purple', ax=ax4)
    ax4.set_title('Log Pi', size=12, pad=10)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Value')

    # 5. Q Value
    ax5 = fig.add_subplot(gs[1, 2])
    sns.lineplot(data=metrics_df, x='step', y='q_value', color='purple', ax=ax5)
    ax5.set_title('Q Values', size=12, pad=10)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Value')

    # 6. beta
    ax6 = fig.add_subplot(gs[1, 3])
    sns.lineplot(data=metrics_df, x='step', y='beta', color='yellow', ax=ax6)
    ax6.set_title('beta', size=12, pad=10)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Value')

    # 7. Training Progress Summary
    ax7 = fig.add_subplot(gs[1, 4])
    # 학습 진행 상황 요약 통계
    total_arrivals = rewards_df['arrived'].sum()
    arrival_rate = (total_arrivals / len(rewards_df)) * 100
    recent_arrival_rate = rewards_df['arrived'].tail(window_size).mean() * 100
    summary_text = (
        f"Total Episodes: {len(rewards_df)}\n"
        f"Max Reward: {rewards_df['reward'].max():.2f}\n"
        f"Min Reward: {rewards_df['reward'].min():.2f}\n"
        f"Avg Reward: {rewards_df['reward'].mean():.2f}\n"
        f"Final Reward: {rewards_df['reward'].iloc[-1]:.2f}\n"
        f"Total Arrivals: {total_arrivals}\n"
        f"Overall Success Rate: {arrival_rate:.1f}%\n"
        f"Recent Success Rate: {recent_arrival_rate:.1f}%\n\n"
        f"Final Actor Loss: {metrics_df['actor_loss'].iloc[-1]:.4f}\n"
        f"Avg Actor Loss: {metrics_df['actor_loss'].mean():.2f}\n"
        f"Final Critic1 Loss: {metrics_df['critic_loss1'].iloc[-1]:.4f}\n"
        f"Final Critic2 Loss: {metrics_df['critic_loss2'].iloc[-1]:.4f}\n"
        f"Avg Critic1 Loss: {metrics_df['critic_loss1'].mean():.2f}\n"
        f"Avg Critic2 Loss: {metrics_df['critic_loss2'].mean():.2f}\n"
        f"Final beta: {metrics_df['beta'].iloc[-1]:.4f}"
    )
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
             verticalalignment='top', fontsize=10)
    ax7.set_title('Training Summary', size=12, pad=10)
    ax7.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("현재 디렉토리 절대 경로:", os.getcwd())
    result_dir = os.getcwd() + "/results/environment"
    plot_training_results(result_dir)
