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

// 디바이스 설정 함수
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


int main(int argc, char* argv[]) {


	return 0;
}