#include "utils/types.hpp"
#include "utils/constants.hpp"
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
        // 드라이버 힌트 설정
        SDL_SetHint(SDL_HINT_RENDER_DRIVER, "direct3d11");  // Windows에서 Direct3D 우선 사용
        SDL_SetHint(SDL_HINT_RENDER_LOGICAL_SIZE_MODE, "0");
        SDL_SetHint(SDL_HINT_RENDER_BATCHING, "1");       // 배치 렌더링 활성화

        window_ = SDL_CreateWindow(
            "GPU Accelerated Object Test",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            Display::WIDTH, Display::HEIGHT,
            SDL_WINDOW_SHOWN
        );

        if (!window_) {
            throw std::runtime_error(std::string("Window creation failed: ") + SDL_GetError());
        }

        SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");

        // GPU 렌더러 생성
        renderer_ = SDL_CreateRenderer(
            window_, -1,
            SDL_RENDERER_ACCELERATED    // GPU 가속 활성화
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
            std::cout << "Max texture size: " << info.max_texture_width << "x" << info.max_texture_height << std::endl;
            std::cout << "Flags:" << std::endl;
            if (info.flags & SDL_RENDERER_SOFTWARE) std::cout << "- Software rendering" << std::endl;
            if (info.flags & SDL_RENDERER_ACCELERATED) std::cout << "- Hardware accelerated" << std::endl;
            if (info.flags & SDL_RENDERER_PRESENTVSYNC) std::cout << "- VSync enabled" << std::endl;
            if (info.flags & SDL_RENDERER_TARGETTEXTURE) std::cout << "- Target texture supported" << std::endl;
        }

        SDL_DisplayMode current;
        if (SDL_GetCurrentDisplayMode(0, &current) == 0) {
            std::cout << "Current Display Mode:\n";
            std::cout << "Width: " << current.w << "\n";
            std::cout << "Height: " << current.h << "\n";
            std::cout << "Refresh Rate: " << current.refresh_rate << "Hz\n";
            std::cout << "========================\n\n";
        }

        // 렌더러 성능 최적화 설정
        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");  // Nearest pixel sampling
        SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);

        // Windows에서 추가 성능 최적화
#ifdef _WIN32
        // DPI 인식 비활성화로 스케일링 오버헤드 감소
        SDL_SetHint(SDL_HINT_WINDOWS_DPI_AWARENESS, "0");
#endif
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

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    std::vector<std::unique_ptr<object::CircleObstacle>> circle_obstacles;
    circle_obstacles.reserve(constants::CircleObstacle::COUNT);
    for (size_t i = 0; i < constants::CircleObstacle::COUNT; i++) {
        auto obs = std::make_unique<object::CircleObstacle>(std::nullopt, std::nullopt, constants::CircleObstacle::RADIUS, constants::CircleObstacle::SPAWN_BOUNDS, Display::to_sdl_color(Display::ORANGE), true);
        obs->reset();  // 초기 위치 설정
        circle_obstacles.push_back(std::move(obs));
    }
    tensor_t circle_obstacles_state = torch::zeros({constants::CircleObstacle::COUNT, 3});
    auto updateCircleObstaclesState = [&circle_obstacles, &circle_obstacles_state]() {
        for (size_t i = 0; i < circle_obstacles.size(); ++i) {
            circle_obstacles_state[i] = circle_obstacles[i]->get_state();
        }
        return circle_obstacles_state;
    };
    circle_obstacles_state = updateCircleObstaclesState();

    std::vector<std::unique_ptr<object::RectangleObstacle>> rectangle_obstacles;
    rectangle_obstacles.reserve(constants::RectangleObstacle::COUNT);
    for (size_t i = 0; i < constants::RectangleObstacle::COUNT; i++) {
        auto obs = std::make_unique<object::RectangleObstacle>(std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, constants::RectangleObstacle::SPAWN_BOUNDS, Display::to_sdl_color(Display::ORANGE), false);
        obs->reset();  // 초기 위치 설정
        rectangle_obstacles.push_back(std::move(obs));
    }
    tensor_t rectangle_obstacles_state = torch::zeros({constants::RectangleObstacle::COUNT, 5});
    auto updateRectangleObstaclesState = [&rectangle_obstacles, &rectangle_obstacles_state]() {
        for (size_t i = 0; i < rectangle_obstacles.size(); ++i) {
            rectangle_obstacles_state[i] = rectangle_obstacles[i]->get_state();
        }
        return rectangle_obstacles_state;
    };
    rectangle_obstacles_state = updateRectangleObstaclesState();

    // 목표 생성 및 초기화
    auto goal = std::make_unique<object::Goal>(500.0f, 50.0f, constants::Goal::RADIUS, constants::Goal::SPAWN_BOUNDS, Display::to_sdl_color(Display::GREEN), false);
    goal->reset();

    // 에이전트 생성 및 초기화
    auto agent = std::make_unique<object::Agent>(500.0f, 950.0f, constants::Agent::RADIUS, constants::Agent::SPAWN_BOUNDS, constants::Agent::MOVE_BOUNDS, Display::to_sdl_color(Display::BLUE), true, circle_obstacles_state, rectangle_obstacles_state, goal->get_state());

    bool quit = false;
    SDL_Event event;

    const real_t dt = 1.0f / Display::FPS;

    const tensor_t forward_action = torch::tensor({0.6f, 0.0f});

    Uint32 frameCount = 0;
    Uint32 lastTime = SDL_GetTicks();
    Uint32 currentTime;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT ||
                (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
                quit = true;
            }
            else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_r) {
                for (auto& obs : circle_obstacles) {
                    obs->reset();
                }
                for (auto& obs : rectangle_obstacles) {
                    obs->reset();
                }
                goal->reset(500.0f, 50.0f);
                updateCircleObstaclesState();
                updateRectangleObstaclesState();
                agent->reset(500.0f, 950.0f, circle_obstacles_state, rectangle_obstacles_state, goal->get_state());
            }
        }

        // 장애물 업데이트 - 고정된 dt로 시뮬레이션
        for (auto& obs : circle_obstacles) {
            obs->update(dt);
        }

        updateCircleObstaclesState();
        agent->update(dt, forward_action, circle_obstacles_state);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        auto color = Display::to_sdl_color(Display::GREEN);
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
        SDL_RenderDrawLine(renderer, 0, Section::GOAL_LINE, Display::WIDTH, Section::GOAL_LINE);
        SDL_RenderDrawLine(renderer, 0, Section::START_LINE, Display::WIDTH, Section::START_LINE);

        for (const auto& obs : circle_obstacles) {
            obs->draw(renderer);
        }

        for (const auto& obs : rectangle_obstacles) {
            obs->draw(renderer);
        }

        goal->draw(renderer);

        agent->draw(renderer);

        SDL_RenderPresent(renderer);

        frameCount++;
        currentTime = SDL_GetTicks();
        if (currentTime > lastTime + 1000) {
            std::cout << "FPS: " << frameCount << std::endl;
            frameCount = 0;
            lastTime = currentTime;
        }
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

int main(int argc, char* argv[]) {

    test_sdl_object();

	return 0;
}