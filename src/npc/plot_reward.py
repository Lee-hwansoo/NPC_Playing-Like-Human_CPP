import numpy as np
import matplotlib.pyplot as plt

# 보상 계산 함수들
def calculate_distance_reward(normalized_goal_dist, dist_factor=0.4):
    """
    거리 기반 보상 계산
    - 1~0.1: 전체 보상의 70% (완만한 증가)
    - 0.1~0: 나머지 30% (급격한 증가)
    """
    # if normalized_goal_dist > 0.1:
    #     # 1~0.1 구간: 70%의 보상을 완만하게 분배
    #     reward = 0.6 * (1 - (normalized_goal_dist - 0.1) / 0.9)
    # else:
    #     # 0.1~0 구간: 나머지 30%의 보상을 급격하게 분배
    #     # 이차함수를 사용하여 0에 가까워질수록 더 가파른 증가
    #     base_reward = 0.6  # 0.1 지점에서의 보상
    #     remaining_reward = 0.4  # 남은 30%의 보상
    #     progress = 1 - (normalized_goal_dist / 0.1)  # 0.1에서 0까지의 진행도
    #     reward = base_reward + remaining_reward * (progress ** 2)

    # if normalized_goal_dist > 0.1:
    #     # 1~0.1 구간
    #     progress = 1 - normalized_goal_dist  # 직접적인 거리 사용
    #     reward = 0.6 * (progress ** 1.5)  # 40%까지만 보상
    # else:
    #     # 0.1~0 구간: 급격한 보상 증가
    #     progress = 1 - (normalized_goal_dist / 0.1)
    #     reward = 0.6 + 0.4 * (progress ** 2.5)  # 0.1에서 60% 시작, 나머지 40% 급격히 증가

    if normalized_goal_dist > 0.1:
        # 1~0.1 구간: 완만한 선형 증가
        progress = 1 - normalized_goal_dist
        reward = 0.8 * (progress / 0.9)  # 80%까지 선형 증가
    else:
        # 0.1~0 구간: 가파른 선형 증가
        near_goal_progress = (0.1 - normalized_goal_dist) / 0.1
        reward = 0.8 + 0.2 * near_goal_progress  # 0.1에서 80% 시작, 선형적으로 100%까지 증가

    return reward * dist_factor

def calculate_path_reward(normalized_frenet_d, path_factor=0.5):
    """
    경로 중심 거리 보상 계산
    """
    reward = np.exp(-np.abs(normalized_frenet_d) * 10.0) * path_factor
    return reward

def calculate_alignment_reward(normalized_alignment, alignment_factor=0.1):
    """
    정렬 기반 보상 계산
    """
    reward = np.exp(-(1.0 - normalized_alignment) * 3.0) * alignment_factor
    return reward

def calculate_stop_penalty(force, stop_factor=0.1):
    """
    정지 패널티 계산
    """
    if force >= 0.1:
        return 0.0
    else:
        penalty = np.exp(-force / 0.1) * stop_factor
    return penalty

# 통합 보상 계산
def calculate_total_reward(normalized_goal_dist, normalized_frenet_d, normalized_alignment,
                           dist_factor=0.4, path_factor=0.5, alignment_factor=0.1):
    dist_reward = calculate_distance_reward(normalized_goal_dist, dist_factor)
    path_reward = calculate_path_reward(normalized_frenet_d, path_factor)
    alignment_reward = calculate_alignment_reward(normalized_alignment, alignment_factor)
    total_reward = dist_reward + path_reward + alignment_reward
    return np.clip(total_reward, 0.0, 1.0)

# 시각화
def visualize_rewards():
    # 거리 보상 그래프
    x_dist = np.linspace(0, 1, 1000)
    y_dist = [calculate_distance_reward(d) for d in x_dist]

    # 경로 중심 보상 그래프
    x_path = np.linspace(-1, 1, 1000)
    y_path = [calculate_path_reward(d) for d in x_path]

    # 정렬 보상 그래프
    x_align = np.linspace(0, 1, 1000)
    y_align = [calculate_alignment_reward(d) for d in x_align]

    # 속도 페널티 그래프
    x_force = np.linspace(0, 1, 1000)
    y_force = [calculate_stop_penalty(d) for d in x_force]

    # 전체 보상 시각화
    plt.figure(figsize=(15, 12))

    plt.subplot(4, 1, 1)
    plt.plot(x_dist, y_dist, 'b-', label='Distance Reward', linewidth=2)
    plt.axvline(x=0.1, color='r', linestyle='--', label='Transition Point')
    plt.title('Distance Reward')
    plt.xlabel('Normalized Goal Distance')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(x_path, y_path, 'g-', label='Path Reward', linewidth=2)
    plt.title('Path Reward')
    plt.xlabel('Normalized Frenet d')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(x_align, y_align, 'm-', label='Alignment Reward', linewidth=2)
    plt.title('Alignment Reward')
    plt.xlabel('Normalized Alignment')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(x_force, y_force, 'r-', label='Stop Penalty', linewidth=2)
    plt.title('Stop Penalty')
    plt.xlabel('Normalized Force')
    plt.ylabel('Penalty')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 테스트 및 분석
def analyze_rewards1():
    distances = np.linspace(0, 1, 1000)
    frenet_distances = np.linspace(-1, 1, 10)
    alignments = np.linspace(0, 1, 10)

    print("Distance Reward Analysis:")
    for d in distances:
        print(f"Distance: {d:.2f}, Reward: {calculate_distance_reward(d):.6f}")

    print("\nPath Reward Analysis:")
    for d in frenet_distances:
        print(f"Frenet d: {d:.2f}, Reward: {calculate_path_reward(d):.6f}")

    print("\nAlignment Reward Analysis:")
    for a in alignments:
        print(f"Alignment: {a:.2f}, Reward: {calculate_alignment_reward(a):.6f}")

def analyze_rewards2():
    distances = {
        "근거리": [0.001, 0.002, 0.003, 0.008, 0.015, 0.02, 0.025, 0.03, 0.035],
        "전환점 근처": [0.08, 0.09, 0.098, 0.099, 0.1, 0.101, 0.102, 0.11, 0.12],
        "원거리": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    }

    for region, points in distances.items():
        print(f"\n{region} 분석:")
        print("거리      | 보상     | 차이")
        print("-" * 40)
        prev_reward = None
        for d in points:
            reward = calculate_distance_reward(d)
            diff = reward - prev_reward if prev_reward is not None else None
            diff_str = f"{diff:+.6f}" if diff is not None else "---"
            print(f"{d:.6f} | {reward:.6f} | {diff_str}")
            prev_reward = reward

    # 변화율 분석
    print("\n변화율 분석:")
    close_points = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.1, 0.3, 0.5, 1.0]
    for i in range(len(close_points)-1):
        d1, d2 = close_points[i], close_points[i+1]
        r1 = calculate_distance_reward(d1)
        r2 = calculate_distance_reward(d2)
        rate = (r2 - r1) / (d2 - d1)
        print(f"구간 {d1:.3f}->{d2:.3f}: 변화율 = {rate:.6f}")

# 실행
if __name__ == "__main__":
    visualize_rewards()
    analyze_rewards1()
    analyze_rewards2()
