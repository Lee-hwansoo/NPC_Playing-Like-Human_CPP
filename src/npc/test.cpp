#include "npc/constants.hpp"
#include "npc/actor.hpp"
#include "npc/critic.hpp"
#include "npc/sac.hpp"

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

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

void test_actor(){
    try{
        std::cout << "\nStarting test..." << std::endl;
        torch::Device device = get_device();
        std::cout << "Device initialized" << std::endl;

        const int64_t state_dim = 159;
        const int64_t action_dim = 2;
        std::vector<float> min_action = {0.6f, -1.0f};
        std::vector<float> max_action = {1.0f, 1.0f};

        std::cout << "Creating Actor network..." << std::endl;
        Actor actor("actor", state_dim, action_dim, min_action, max_action);
        actor->to(device);

        std::cout << "Successfully created " << actor->network_name() << " network " << std::endl;

        actor->load_network_parameters("20241024_002829", 1800);
        // actor->save_network_parameters(1801);

        std::cout << "\nTesting single state..." << std::endl;
        auto state = torch::randn({1, state_dim});
        std::cout << "Input state shape: " << state.sizes() << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        auto [action, log_prob] = actor->sample(state);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "Output action shape: " << action.sizes() << std::endl;
        std::cout << "Action: " << action << std::endl;
        std::cout << "Log probability: " << log_prob << std::endl;
        std::cout << "\nAll tests completed successfully! Execution time: " << elapsed.count() << " ms\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}

void test_critic(){
    try{
        std::cout << "\nStarting test..." << std::endl;
        torch::Device device = get_device();
        std::cout << "Device initialized" << std::endl;

        const int64_t state_dim = 159;
        const int64_t action_dim = 2;

        std::cout << "Creating Critic network..." << std::endl;
        Critic critic("critic1", state_dim, action_dim);
        critic->to(device);

        std::cout << "Successfully created " << critic->network_name() << " network " << std::endl;

        critic->load_network_parameters("20241024_002829", 1800);
        // critic->save_network_parameters(1801);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}

torch::Tensor get_random_state(int64_t state_dim, torch::Device device) {
    return torch::randn({1, state_dim}).to(device);
}

torch::Tensor get_reward(const torch::Tensor& state, const torch::Tensor& action) {
    return torch::exp(-torch::norm(state) - torch::norm(action));
}

bool is_done(const torch::Tensor& state) {
    return torch::norm(state).item<float>() > 5.0f;
}

void test_sac(){
    try{
        std::cout << "\nStarting test..." << std::endl;
        torch::Device device = get_device();
        std::cout << "Device initialized" << std::endl;

        const int64_t state_dim = 159;
        const int64_t action_dim = 2;
        const std::vector<float> min_action = {0.6f, -1.0f};
        const std::vector<float> max_action = {1.0f, 1.0f};

        SAC sac(state_dim, action_dim, min_action, max_action, device);

        sac.load_network_parameters("20241024_002829", 1800);

        const int episodes = 100;
        const int max_steps = 100;

        std::cout << "\nStarting training loop...\n" << std::endl;
        sac.train();
        for (int episode = 0; episode < episodes; ++episode) {
            auto state = get_random_state(state_dim, device);
            float total_reward = 0.0f;

            for (int step = 0; step < max_steps; ++step) {
                auto action = sac.select_action(state);

                auto next_state = get_random_state(state_dim, device);
                auto reward = get_reward(state, action);
                auto done = is_done(next_state);

                sac.add(
                    state,
                    action,
                    reward.view({1, 1}),
                    next_state,
                    torch::tensor(done ? 1.0f : 0.0f).view({1, 1}).to(device)
                );

                sac.update();

                total_reward += reward.item<float>();
                state = next_state;

                if (done) {
                    break;
                }
            }

            std::cout << "Episode " << episode + 1
                    << ", Total Reward: " << total_reward
                    << std::endl;
        }

        std::cout << "\nTraining completed!" << std::endl;

        std::cout << "\nTesting the trained model...\n" << std::endl;
        sac.eval();
        auto test_state = get_random_state(state_dim, device);
        std::cout << "Test state: " << test_state << std::endl;

        const int num_tests = 1000;
        std::vector<double> execution_times;
        execution_times.reserve(num_tests);
        std::cout << "\nMeasuring select_action performance over " << num_tests << " runs...\n" << std::endl;

        double total_time = 0.0;
        for (int i = 0; i < num_tests; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto test_action = sac.select_action(test_state);
            auto end = std::chrono::high_resolution_clock::now();

            auto test_reward = get_reward(test_state, test_action);
            std::cout << "Received reward: " << test_reward.item<float>() << std::endl;

            std::chrono::duration<double, std::milli> elapsed = end - start;
            execution_times.push_back(elapsed.count());
            total_time += elapsed.count();
        }

        double mean_time = total_time / num_tests;
        double sq_sum = 0.0;
        for (double time : execution_times) {
            sq_sum += (time - mean_time) * (time - mean_time);
        }
        double std_dev = std::sqrt(sq_sum / num_tests);
        auto [min_time, max_time] = std::minmax_element(execution_times.begin(), execution_times.end());

        std::cout << "\nSelect Action Performance Stats (" << num_tests << " runs):" << std::endl;
        std::cout << "  Average time: " << mean_time << " ms" << std::endl;
        std::cout << "  Std Dev: " << std_dev << " ms" << std::endl;
        std::cout << "  Min time: " << *min_time << " ms" << std::endl;
        std::cout << "  Max time: " << *max_time << " ms" << std::endl;

        std::cout << "\nAll tests completed successfully!" << std::endl;

        sac.print_model_info();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}

int main(){
    // test_actor();
    // test_critic();

    test_sac();
    return 0;
}