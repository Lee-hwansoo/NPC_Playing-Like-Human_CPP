#include <npc/example.hpp>
#include <SDL.h>
#include <Eigen/Dense>
#include <cstdint>
#include <iostream>
#include <cassert>
#include <chrono>

void testCharacter() {
  example::Character hero("Hero", 100);

  std::cout << "Testing Character class:\n";
  std::cout << "Name: " << hero.getName() << ", Health: " << hero.getHealth() << "\n";

  hero.takeDamage(20);
  std::cout << "After taking 20 damage, Health: " << hero.getHealth() << "\n";

  hero.heal(10);
  std::cout << "After healing 10, Health: " << hero.getHealth() << "\n";

  assert(hero.getName() == "Hero");
  assert(hero.getHealth() == 90);
}

void testCalculateDamage() {
    std::cout << "\nTesting calculateDamage function:\n";
    int damage = example::calculateDamage(10, 5);
    std::cout << "Calculated damage (base 10, level 5): " << damage << "\n";
    assert(damage > 10); // Assuming the function increases damage based on level
}

void testSplitString() {
  std::cout << "\nTesting splitString function:\n";
  std::string test = "Hello,World,OpenAI";
  std::vector<std::string> result = example::splitString(test, ',');

  std::cout << "Split result: ";
  for (const auto& str : result) {
      std::cout << str << " ";
  }
  std::cout << "\n";

  assert(result.size() == 3);
  assert(result[0] == "Hello");
  assert(result[1] == "World");
  assert(result[2] == "OpenAI");
}

void testClamp() {
  std::cout << "\nTesting clamp function:\n";

  int intResult = example::clamp(15, 0, 10);
  std::cout << "Clamp int (15, 0, 10): " << intResult << "\n";
  assert(intResult == 10);

  float floatResult = example::clamp(3.14f, 0.0f, 1.0f);
  std::cout << "Clamp float (3.14, 0.0, 1.0): " << floatResult << "\n";
  assert(floatResult == 1.0f);
}

void testSDL2() {
    std::cout << "\nTesting SDL2 initialization and window creation:\n";

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        assert(false);
        return;
    }

    SDL_Window* window = SDL_CreateWindow("SDL2 Test Window", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 640, 480, SDL_WINDOW_SHOWN);
    if (window == nullptr) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        assert(false);
        SDL_Quit();
        return;
    }

    SDL_Surface* screenSurface = SDL_GetWindowSurface(window);
    SDL_FillRect(screenSurface, nullptr, SDL_MapRGB(screenSurface->format, 0xFF, 0xFF, 0xFF));
    SDL_UpdateWindowSurface(window);

    SDL_Event e;
    bool quit = false;
    Uint32 startTime = SDL_GetTicks();

    while (!quit && SDL_GetTicks() - startTime < 2000) {  // Run for 2 seconds
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }
    }

    SDL_DestroyWindow(window);
    SDL_Quit();

    std::cout << "SDL2 test completed successfully.\n";
}

void testEigen3() {
    std::cout << "\nTesting Eigen3 matrix operations:\n";

    // 3x3 행렬 생성
    Eigen::Matrix3d matrix;
    matrix << 1, 2, 3,
              4, 5, 6,
              7, 8, 9;

    std::cout << "Original matrix:\n" << matrix << "\n\n";

    // 행렬의 전치 계산
    Eigen::Matrix3d transposed = matrix.transpose();
    std::cout << "Transposed matrix:\n" << transposed << "\n\n";

    // 행렬의 역행렬 계산
    Eigen::Matrix3d inverse = matrix.inverse();
    std::cout << "Inverse matrix:\n" << inverse << "\n\n";

    // 행렬과 그 역행렬의 곱 계산 (단위 행렬이 나와야 함)
    Eigen::Matrix3d identity = matrix * inverse;
    std::cout << "Matrix multiplied by its inverse (should be identity):\n" << identity << "\n\n";

    // 단위 행렬인지 확인
    assert((identity - Eigen::Matrix3d::Identity()).norm() < 1e-12);

    // 고유값 계산
    Eigen::EigenSolver<Eigen::Matrix3d> solver(matrix);
    std::cout << "Eigenvalues:\n" << solver.eigenvalues() << "\n";

    std::cout << "Eigen3 test completed successfully.\n";
}

int32_t main() {
  std::cout << "Starting tests for NPC project.\n\n";

  auto start = std::chrono::high_resolution_clock::now();
  testCharacter();
  testCalculateDamage();
  testSplitString();
  testClamp();
  testSDL2();
  // testEigen3();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "\nAll tests completed successfully! Execution time: " << elapsed.count() << " ms\n";
  return 0;
}