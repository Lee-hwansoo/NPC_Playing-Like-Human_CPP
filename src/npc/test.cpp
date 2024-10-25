#include "utils/types.hpp"
#include "utils/constants.hpp"
#include "utils/utils.hpp"
#include "npc/actor.hpp"
#include "npc/critic.hpp"
#include "npc/sac.hpp"
#include "npc/object.hpp"
#include <SDL.h>
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

void test_actor(){
    try{
        std::cout << "\nStarting test..." << std::endl;
        torch::Device device = get_device();
        std::cout << "Device initialized" << std::endl;

        const dim_type state_dim = 159;
        const dim_type action_dim = 2;
        std::vector<real_t> min_action = {0.6f, -1.0f};
        std::vector<real_t> max_action = {1.0f, 1.0f};

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
        std::chrono::duration<real_real_t, std::milli> elapsed = end - start;

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

        const dim_type state_dim = 159;
        const dim_type action_dim = 2;

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

tensor_t get_random_state(dim_type state_dim, torch::Device device) {
    return torch::randn({1, state_dim}).to(device);
}

tensor_t get_reward(const tensor_t& state, const tensor_t& action) {
    return torch::exp(-torch::norm(state) - torch::norm(action));
}

bool is_done(const tensor_t& state) {
    return torch::norm(state).item<real_t>() > 5.0f;
}

void test_sac(){
    try{
        std::cout << "\nStarting test..." << std::endl;
        torch::Device device = get_device();
        std::cout << "Device initialized" << std::endl;

        const dim_type state_dim = 159;
        const dim_type action_dim = 2;
        const std::vector<real_t> min_action = {0.6f, -1.0f};
        const std::vector<real_t> max_action = {1.0f, 1.0f};

        SAC sac(state_dim, action_dim, min_action, max_action, device);

        sac.load_network_parameters("20241024_002829", 1800);

        const int episodes = 100;
        const int max_steps = 100;

        std::cout << "\nStarting training loop...\n" << std::endl;
        sac.train();
        for (int episode = 0; episode < episodes; ++episode) {
            auto state = get_random_state(state_dim, device);
            real_t total_reward = 0.0f;

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

                total_reward += reward.item<real_t>();
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
        std::vector<real_real_t> execution_times;
        execution_times.reserve(num_tests);
        std::cout << "\nMeasuring select_action performance over " << num_tests << " runs...\n" << std::endl;

        real_real_t total_time = 0.0;
        for (int i = 0; i < num_tests; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto test_action = sac.select_action(test_state);
            auto end = std::chrono::high_resolution_clock::now();

            auto test_reward = get_reward(test_state, test_action);
            std::cout << "Received reward: " << test_reward.item<real_t>() << std::endl;

            std::chrono::duration<real_real_t, std::milli> elapsed = end - start;
            execution_times.push_back(elapsed.count());
            total_time += elapsed.count();
        }

        real_real_t mean_time = total_time / num_tests;
        real_real_t sq_sum = 0.0;
        for (real_real_t time : execution_times) {
            sq_sum += (time - mean_time) * (time - mean_time);
        }
        real_real_t std_dev = std::sqrt(sq_sum / num_tests);
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

void test_frenet(){
    try{
        std::cout << "\nStarting test..." << std::endl;
        torch::Device device = get_device();
        std::cout << "Device initialized" << std::endl;

        tensor_t position = torch::tensor({1.0, 2.0}, get_tensor_dtype()).to(device);
        tensor_t path = torch::tensor({{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}}, get_tensor_dtype()).to(device);

        auto result = utils::FrenetCoordinate::getFrenetD(position, path, device);

        if (result.has_value()) {
            auto [closest_point, lateral_distance] = result.value();
            std::cout << "Closest point: \n" << closest_point << std::endl;
            std::cout << "Lateral distance: " << lateral_distance << std::endl;
        } else {
            std::cout << "Path has less than 2 waypoints" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}

class SDLWrapper {
public:
    SDLWrapper() {
        // GPU 가속을 위한 SDL 초기화
        if (SDL_InitSubSystem(SDL_INIT_VIDEO) < 0) {
            throw std::runtime_error(std::string("SDL initialization failed: ") + SDL_GetError());
        }

        // OpenGL 설정
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    }

    ~SDLWrapper() {
        SDL_Quit();
    }
};

class RenderWindow {
public:
    RenderWindow() {
        // OpenGL 지원과 함께 윈도우 생성
        window_ = SDL_CreateWindow(
            "GPU Accelerated Object Test",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            Display::WIDTH, Display::HEIGHT,
            SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL
        );

        if (!window_) {
            throw std::runtime_error(std::string("Window creation failed: ") + SDL_GetError());
        }

        // GPU 가속 렌더러 생성
        renderer_ = SDL_CreateRenderer(
            window_, -1,
            SDL_RENDERER_ACCELERATED |    // GPU 가속 활성화
            SDL_RENDERER_PRESENTVSYNC |   // 수직동기화 활성화
            SDL_RENDERER_TARGETTEXTURE    // 렌더 타겟 텍스처 지원
        );

        if (!renderer_) {
            SDL_DestroyWindow(window_);
            throw std::runtime_error(std::string("Renderer creation failed: ") + SDL_GetError());
        }

        // 렌더러 정보 출력
        SDL_RendererInfo info;
        if (SDL_GetRendererInfo(renderer_, &info) == 0) {
            std::cout << "Renderer information:" << std::endl;
            std::cout << "Name: " << info.name << std::endl;
            std::cout << "Flags:" << std::endl;
            if (info.flags & SDL_RENDERER_SOFTWARE) std::cout << "- Software rendering" << std::endl;
            if (info.flags & SDL_RENDERER_ACCELERATED) std::cout << "- Hardware accelerated" << std::endl;
            if (info.flags & SDL_RENDERER_PRESENTVSYNC) std::cout << "- VSync enabled" << std::endl;
            if (info.flags & SDL_RENDERER_TARGETTEXTURE) std::cout << "- Target texture supported" << std::endl;
        }

        // 렌더러 품질 설정
        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");  // 선형 필터링

        // 블렌딩 모드 설정
        SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
    }

    ~RenderWindow() {
        if (renderer_) SDL_DestroyRenderer(renderer_);
        if (window_) SDL_DestroyWindow(window_);
    }

    SDL_Renderer* getRenderer() { return renderer_; }
    SDL_Window* getWindow() { return window_; }

private:
    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
};

void testIntegratedObjects(SDL_Renderer* renderer) {
    std::cout << "Starting simple object display test...\n";

    // 장애물 생성
    std::vector<std::unique_ptr<object::CircleObstacle>> obstacles;
    obstacles.reserve(Obstacle::COUNT);
    for (size_t i = 0; i < Obstacle::COUNT; i++) {
        auto obs = std::make_unique<object::CircleObstacle>();
        obs->reset();  // 초기 위치 설정
        obstacles.push_back(std::move(obs));
    }

    tensor_t obstacles_state = torch::zeros({Obstacle::COUNT, 3}, get_tensor_dtype());

    auto updateObstaclesState = [&obstacles, &obstacles_state]() {
        for (size_t i = 0; i < obstacles.size(); ++i) {
            obstacles_state[i] = obstacles[i]->get_state();
        }
        return obstacles_state;
    };

    obstacles_state = updateObstaclesState();

    // 목표 생성 및 초기화
    auto goal = std::make_unique<object::Goal>();
    goal->reset();

    // 에이전트 생성 및 초기화
    auto agent = std::make_unique<object::Agent>(500.0f, 900.0f, 10.0f, constants::Agent::BOUNDS, Display::to_sdl_color(Display::BLUE), true, obstacles_state, goal->get_state());

    bool quit = false;
    SDL_Event event;

    // 시뮬레이션 시간 간격은 고정
    const real_t dt = 1.0f / Display::FPS;  // 물리 시뮬레이션용 고정 시간 간격

    const tensor_t forward_action = torch::tensor({0.8f, 0.0f});

    while (!quit) {
        // 이벤트 처리
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT ||
                (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
                quit = true;
            }
            else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_r) {
                for (auto& obs : obstacles) {
                    obs->reset();
                }
                goal->reset();
                updateObstaclesState();
                agent->reset(500.0f, 900.0f, obstacles_state, goal->get_state());
            }
        }

        // 장애물 업데이트 - 고정된 dt로 시뮬레이션
        for (auto& obs : obstacles) {
            obs->update(dt);
        }

        updateObstaclesState();
        agent->update(dt, forward_action, obstacles_state, goal->get_state());

        // 화면 클리어 및 렌더링 - 가능한 빠르게 처리
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        auto color = Display::to_sdl_color(Display::GREEN);
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
        SDL_RenderDrawLine(renderer, 0, Section::GOAL_LINE, Display::WIDTH, Section::GOAL_LINE);
        SDL_RenderDrawLine(renderer, 0, Section::START_LINE, Display::WIDTH, Section::START_LINE);

        for (const auto& obs : obstacles) {
            obs->draw(renderer);
        }

        goal->draw(renderer);

        agent->draw(renderer);

        SDL_RenderPresent(renderer);
    }

    std::cout << "Display test completed.\n";
}

int test_sdl_object(){
    try {
        SDLWrapper sdl;
        RenderWindow window;

        testIntegratedObjects(window.getRenderer());
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char* argv[]){
    // test_actor();
    // test_critic();
    // test_sac();
    // test_frenet();
    // test_sdl_object();
    return 0;
}