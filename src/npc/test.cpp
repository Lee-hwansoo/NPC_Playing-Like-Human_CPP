#include "utils/constants.hpp"
#include "npc/environment.hpp"
#include <SDL.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>

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

        // 비디오 시스템 초기화 (최소 오버헤드)
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            throw std::runtime_error(std::string("SDL init failed: ") + SDL_GetError());
        }

        // OpenGL 설정 - 고성능 3D 렌더링을 위한 설정
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);        // 더블 버퍼링으로 티어링 방지
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);         // 24비트 깊이 버퍼
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);  // 안티앨리어싱 활성화
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);  // 4x MSAA

        // 전역 성능 최적화 설정
        SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0");           // VSync 비활성화로 최대 FPS
        SDL_SetHint(SDL_HINT_RENDER_BATCHING, "1");        // 배치 렌더링으로 드로우콜 최소화
        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");   // 최근접 필터링으로 GPU 부하 감소
        SDL_SetHint(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS, "0"); // 백그라운드 실행 유지

        std::cout << "SDL initialization completed - Version: "
                  << (int)SDL_MAJOR_VERSION << "."
                  << (int)SDL_MINOR_VERSION << "."
                  << (int)SDL_PATCHLEVEL << "\n";
    }

    ~SDLWrapper() {
        SDL_Quit();
    }
};

class RenderWindow {
public:
    RenderWindow() {
        std::cout << "\nInitializing Render Window...\n";
        std::cout << "Display: " << constants::Display::WIDTH
                  << "x" << constants::Display::HEIGHT << "\n";

        // 플랫폼별 최적화 설정
        #ifdef _WIN32
            std::cout << "Platform: Windows\n";
            SDL_SetHint(SDL_HINT_RENDER_DRIVER, "direct3d11");     // D3D11 하드웨어 가속
            SDL_SetHint(SDL_HINT_WINDOWS_DPI_AWARENESS, "unaware"); // DPI 스케일링 비활성화
            SDL_SetHint(SDL_HINT_RENDER_LOGICAL_SIZE_MODE, "0");   // 논리적 크기 모드 비활성화
        #elif defined(__APPLE__)
            std::cout << "Platform: MacOS\n";
            SDL_SetHint(SDL_HINT_RENDER_DRIVER, "metal");          // Metal API 사용
            SDL_SetHint(SDL_HINT_VIDEO_MAC_FULLSCREEN_SPACES, "0"); // Spaces 전환 비활성화
        #endif

        // 윈도우 생성
        uint32_t windowFlags = SDL_WINDOW_SHOWN;
        #ifdef __APPLE__
            windowFlags |= SDL_WINDOW_METAL;    // MacOS Metal 지원
        #else
            windowFlags |= SDL_WINDOW_OPENGL;   // OpenGL 지원
        #endif

        window_ = SDL_CreateWindow(
            "Pearl Abyss",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            constants::Display::WIDTH, constants::Display::HEIGHT,
            windowFlags
        );

        if (!window_) {
            throw std::runtime_error(std::string("Window creation failed: ") + SDL_GetError());
        }

        // 고성능 렌더러 생성
        renderer_ = SDL_CreateRenderer(
            window_, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE
        );

        if (!renderer_) {
            SDL_DestroyWindow(window_);
            throw std::runtime_error(std::string("Renderer creation failed: ") + SDL_GetError());
        }

        // 렌더러 최적화 설정
        SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);

        // 배경색을 검은색으로 설정하고 화면 클리어
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
        SDL_RenderClear(renderer_);
        SDL_RenderPresent(renderer_);

        // 시스템 정보 출력
        printSystemInfo();
        initializePerformanceSettings();
    }

    ~RenderWindow() {
        if (renderer_) SDL_DestroyRenderer(renderer_);
        if (window_) SDL_DestroyWindow(window_);
    }

    SDL_Renderer* getRenderer() const { return renderer_; }
    SDL_Window* getWindow() const { return window_; }

private:
    void printSystemInfo() {
        // 디스플레이 정보 출력
        SDL_DisplayMode current;
        if (SDL_GetCurrentDisplayMode(0, &current) == 0) {
            std::cout << "\nSystem Display:\n";
            std::cout << "- Resolution: " << current.w << "x" << current.h << "\n";
            std::cout << "- Refresh Rate: " << current.refresh_rate << "Hz\n";
            std::cout << "- Format: " << SDL_GetPixelFormatName(current.format) << "\n";
        }

        // 렌더러 정보 출력
        SDL_RendererInfo info;
        if (SDL_GetRendererInfo(renderer_, &info) == 0) {
            std::cout << "\nRenderer:\n";
            std::cout << "- Name: " << info.name << "\n";
            std::cout << "- Max texture: " << info.max_texture_width
                     << "x" << info.max_texture_height << "\n";

            std::cout << "Features:\n";
            if (info.flags & SDL_RENDERER_SOFTWARE)
                std::cout << "- Software rendering\n";
            if (info.flags & SDL_RENDERER_ACCELERATED)
                std::cout << "- Hardware accelerated\n";
            if (info.flags & SDL_RENDERER_PRESENTVSYNC)
                std::cout << "- VSync enabled\n";
            if (info.flags & SDL_RENDERER_TARGETTEXTURE)
                std::cout << "- Target texture supported\n";
        }
    }

    void initializePerformanceSettings() {
        std::cout << "\nPerformance Settings:\n";

        #ifdef _WIN32
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
            std::cout << "- Windows priority: HIGHEST\n";
        #elif defined(__APPLE__)
            setpriority(PRIO_PROCESS, 0, -20);
            std::cout << "- MacOS priority: MAX (-20)\n";
        #endif

        std::cout << "- VSync: Disabled\n";
        std::cout << "- Batch rendering: Enabled\n";
        std::cout << "- GPU acceleration: Enabled\n";
        std::cout << "========================\n\n";
    }

    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
};

void testBasicTrainEnvironmnet(SDL_Renderer* renderer) {
    environment::TrainEnvironment env(constants::Display::WIDTH, constants::Display::HEIGHT);
    env.set_render(renderer);
    // env.load("20241031_004809", 200);
    env.train(150, true);
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