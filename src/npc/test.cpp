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
            "Pearl Abyss",
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

void testBasicTrainEnvironmnet(SDL_Renderer* renderer) {
    environment::TrainEnvironment env(constants::Display::WIDTH, constants::Display::HEIGHT);
    env.set_render(renderer);
    // env.load("20241030_235546", 100);
    env.train(100, true);
}

int main(int argc, char* argv[]) {
    try {
        SDLWrapper sdl;
        RenderWindow window;

        testBasicTrainEnvironmnet(window.getRenderer());
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}