#include "npc/constants.hpp"
#include "torch/torch.h"

#include <iostream>

torch::Device get_device() {
    torch::Device device(torch::kCPU);

#ifdef _WIN32
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA Device: " << torch::cuda::get_device_name() << std::endl;
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

int main(){
    std::cout << "Hello, World!" << std::endl;
    std::cout << Constants::Display::WIDTH << std::endl;

    torch::Device device = get_device();
    auto tensor = torch::ones({3, 3}).to(device);
    auto model = torch::nn::Linear(10, 5);
    model->to(device);
    auto input = torch::randn({1, 10}).to(device);
    auto target = torch::randn({1, 5}).to(device);
    auto output = model->forward(input);
    auto loss = torch::mse_loss(output, target);
    loss.backward();

    return 0;
}