import numpy as np
import matplotlib.pyplot as plt

def calculate_reward(dist, dist_factor=0.4):
    if dist < 0.05:
        # 근거리: 2차 함수적 증가
        t = 1.0 - (dist/0.05)  # 0->1
        reward = (0.85 + t * t) * dist_factor
    else:
        # 원거리: 지수적 증가
        reward = np.exp(-dist * 4.0) * dist_factor
    return reward

# 다양한 거리에서의 보상값 계산
distances = {
    "근거리": [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
    "전환점 근처": [0.048, 0.049, 0.05, 0.051, 0.052],
    "원거리": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
}

for region, points in distances.items():
    print(f"\n{region} 분석:")
    print("거리      | 보상     | 차이")
    print("-" * 40)
    prev_reward = None
    for d in points:
        reward = calculate_reward(d)
        diff = reward - prev_reward if prev_reward is not None else None
        diff_str = f"{diff:+.6f}" if diff is not None else "---"
        print(f"{d:.6f} | {reward:.6f} | {diff_str}")
        prev_reward = reward

# 그래프 시각화
plt.figure(figsize=(15, 10))

# 전체 범위 보상
x_full = np.linspace(0, 1, 1000)
y_full = [calculate_reward(d) for d in x_full]

plt.subplot(2, 1, 1)
plt.plot(x_full, y_full, 'b-', linewidth=2)
plt.axvline(x=0.05, color='r', linestyle='--', label='Transition point')
plt.grid(True)
plt.title('Full Range Reward Function')
plt.xlabel('normalized_goal_dist')
plt.ylabel('reward')
plt.legend()

# 근거리 상세
x_close = np.linspace(0, 0.1, 1000)
y_close = [calculate_reward(d) for d in x_close]

plt.subplot(2, 1, 2)
plt.plot(x_close, y_close, 'b-', linewidth=2)
plt.axvline(x=0.05, color='r', linestyle='--', label='Transition point')
plt.grid(True)
plt.title('Close Range Detail (0~0.1)')
plt.xlabel('normalized_goal_dist')
plt.ylabel('reward')
plt.legend()

plt.tight_layout()

plt.show()

# 변화율 분석
print("\n변화율 분석 (근거리):")
close_points = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04]
for i in range(len(close_points)-1):
    d1, d2 = close_points[i], close_points[i+1]
    r1 = calculate_reward(d1)
    r2 = calculate_reward(d2)
    rate = (r2 - r1)/(d2 - d1)
    print(f"구간 {d1:.3f}->{d2:.3f}: 변화율 = {rate:.6f}")
