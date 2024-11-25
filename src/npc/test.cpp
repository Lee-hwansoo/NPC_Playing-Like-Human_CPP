#include "utils/constants.hpp"
#include "npc/environment.hpp"
#include <SDL.h>
#include <SDL_render.h>
#include <iostream>
#include <torch/torch.h>

#ifdef _WIN32
#include <windows.h>
#endif

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
        std::cout << "Initializing SDL...\n";

        // 최소한의 비디오 시스템만 초기화
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            throw std::runtime_error(std::string("SDL init failed: ") + SDL_GetError());
        }

        // 최소한의 OpenGL 설정
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);  // OpenGL 버전 낮춤
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);  // 기본적인 더블 버퍼링만 유지

        // 고성능 설정들 제거
        // SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);         // 제거
        // SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);  // 안티앨리어싱 비활성화
        // SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);  // 제거

        // 최소 성능 설정
        SDL_SetHint(SDL_HINT_RENDER_VSYNC, "1");           // VSync 활성화로 GPU 부하 감소
        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");   // 최근접 필터링 유지
        // SDL_SetHint(SDL_HINT_RENDER_BATCHING, "1");     // 제거

        std::cout << "SDL initialization completed\n";
    }

    ~SDLWrapper() {
        SDL_Quit();
    }
};

class RenderWindow {
public:
    RenderWindow() {
        std::cout << "Initializing Render Window...\n";

        // 플랫폼별 설정 최소화
        #ifdef _WIN32
            SDL_SetHint(SDL_HINT_RENDER_DRIVER, "software");     // 소프트웨어 렌더링으로 변경
            SDL_SetHint(SDL_HINT_WINDOWS_DPI_AWARENESS, "unaware"); // DPI 스케일링 비활성화
        #elif defined(__APPLE__)
            SDL_SetHint(SDL_HINT_RENDER_DRIVER, "software");     // 소프트웨어 렌더링으로 변경
        #endif

        // 최소한의 윈도우 플래그
        uint32_t windowFlags = SDL_WINDOW_SHOWN;

        window_ = SDL_CreateWindow(
            "Pearl Abyss",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            constants::Display::WIDTH, constants::Display::HEIGHT,
            windowFlags
        );

        if (!window_) {
            throw std::runtime_error(std::string("Window creation failed: ") + SDL_GetError());
        }

        // 소프트웨어 렌더러 사용
        renderer_ = SDL_CreateRenderer(
            window_, -1,
            SDL_RENDERER_SOFTWARE  // 하드웨어 가속 대신 소프트웨어 렌더링 사용
        );

        if (!renderer_) {
            SDL_DestroyWindow(window_);
            throw std::runtime_error(std::string("Renderer creation failed: ") + SDL_GetError());
        }

        // 최소한의 렌더러 설정
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
        SDL_RenderClear(renderer_);
        SDL_RenderPresent(renderer_);
    }

    ~RenderWindow() {
        if (renderer_) SDL_DestroyRenderer(renderer_);
        if (window_) SDL_DestroyWindow(window_);
    }

    SDL_Renderer* getRenderer() const { return renderer_; }
    SDL_Window* getWindow() const { return window_; }

private:
    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
};

void testBasicTrainEnvironment(SDL_Renderer* renderer) {
    torch::Device device = get_device();
    environment::TrainEnvironment env(constants::Display::WIDTH, constants::Display::HEIGHT, device, 1, true);
    env.set_render(renderer);
    // env.load("20241125_063502", 1200);
    env.train(1000, false, false, false);
    // env.test(10, true);
}

void testMultiAgentEnvironment(SDL_Renderer* renderer) {
    torch::Device device = get_device();
    environment::MultiAgentEnvironment env(constants::Display::WIDTH, constants::Display::HEIGHT, torch::kCPU, 5);
    env.set_render(renderer);
    // env.load("20241112_051733", 5000);
    env.test(true);
}

void testMazeAgentEnvironment(SDL_Renderer* renderer) {
    torch::Device device = get_device();
    environment::MazeAgentEnvironment env(constants::Display::WIDTH, constants::Display::HEIGHT, torch::kCPU, 5);
    env.set_render(renderer);
    // env.load("20241112_051733", 5000);
    env.test(true);
}

int main(int argc, char* argv[]) {
    try {
        SDLWrapper sdl;
        RenderWindow window;

        testBasicTrainEnvironment(window.getRenderer());
        // testMultiAgentEnvironment(window.getRenderer());
        // testMazeAgentEnvironment(window.getRenderer());
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}