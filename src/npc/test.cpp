#include "utils/types.hpp"
#include "utils/constants.hpp"
#include "utils/utils.hpp"
#include "npc/actor.hpp"
#include "npc/critic.hpp"
#include "npc/sac.hpp"
#include "npc/object.hpp"
#include "npc/path_planning.hpp"
#include "npc/environment.hpp"
#include <SDL.h>
#include <memory>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

using namespace object;
using namespace constants;
using namespace types;

torch::Device get_device() {
	torch::Device device(torch::kCPU);

#ifdef _WIN32
	if (torch::cuda::is_available()) {
		device = torch::Device(torch::kCUDA);
		std::cout << "Using CUDA Device" << std::endl;
	}
#elif defined(__APPLE__)
	if (torch::mps::is_available()) {
		device = torch::Device(torch::kMPS);
		std::cout << "Using MPS Device" << std::endl;
	}
#endif
	if (device.type() == torch::kCPU) {
		std::cout << "Using CPU Device" << std::endl;
	}

	return device;
}


void test_training_mode(bool render = false) {
    // SDL 초기화 (렌더링 필요한 경우)
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;

    if (render) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
            return;
        }

        window = SDL_CreateWindow("Training Environment",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            constants::Display::WIDTH, constants::Display::HEIGHT,
            SDL_WINDOW_SHOWN);

        if (!window) {
            std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            return;
        }

        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer) {
            std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window);
            SDL_Quit();
            return;
        }
    }

    try {
        // CUDA 사용 가능한 경우 GPU 사용, 아니면 CPU 사용
        torch::Device device = torch::kCPU;

        // 학습 환경 생성
        TrainEnvironment env(constants::Display::WIDTH, constants::Display::HEIGHT, device);
        if (render) {
            env.set_render(renderer);
        }

        // 훈련 실행
        std::cout << "Starting training..." << std::endl;
        const dim_type training_episodes = 1000;
        auto training_rewards = env.train(training_episodes, render);

        // 훈련 결과 출력
        real_t total_reward = std::accumulate(training_rewards.begin(), training_rewards.end(), 0.0f);
        std::cout << "Training completed. Average reward per episode: "
                  << total_reward / training_episodes << std::endl;

        // 테스트 실행
        std::cout << "Starting testing..." << std::endl;
        const dim_type test_episodes = 100;
        auto test_rewards = env.test(test_episodes, render);

        // 테스트 결과 출력
        total_reward = std::accumulate(test_rewards.begin(), test_rewards.end(), 0.0f);
        std::cout << "Testing completed. Average reward per episode: "
                  << total_reward / test_episodes << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << std::endl;
    }

    // SDL 정리
    if (render) {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
}

int main(int argc, char* argv[]){
    test_training_mode(true);
    return 0;
}