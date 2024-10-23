#include "npc/constants.hpp"
#include "npc/actor.hpp"
#include "torch/torch.h"

#include <iostream>
#include <vector>

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

void test(){
    try{
        std::cout << "Starting test..." << std::endl;
        torch::Device device = get_device();
        std::cout << "Device initialized" << std::endl;

        const int64_t state_dim = 4;
        const int64_t action_dim = 2;
        std::vector<float> min_action = {0.6f, -1.0f};
        std::vector<float> max_action = {1.0f, 1.0f};

        std::cout << "Creating Actor network..." << std::endl;
        Actor actor(state_dim, action_dim, min_action, max_action);
        actor->to(device);

        std::cout << "\nTesting single state..." << std::endl;
        auto state = torch::randn({1, state_dim}).to(device);
        std::cout << "Input state shape: " << state.sizes() << std::endl;

        auto [action, log_prob] = actor->sample(state);
        std::cout << "Output action shape: " << action.sizes() << std::endl;
        std::cout << "Action: " << action << std::endl;
        std::cout << "Log probability: " << log_prob << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}

int main(){
    auto m = torch::nn::Linear(20, 30);
    auto dropout = torch::nn::Dropout(0.1);
    auto input = torch::randn({128, 20});
    auto output = m->forward(input);
    output = dropout->forward(output);
    std::cout << output.sizes() << std::endl;

    test();
    return 0;
}