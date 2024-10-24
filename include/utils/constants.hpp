#pragma once

#include "utils/types.hpp"
#include <SDL.h>
#include <random>

namespace constants
{
    using namespace types;

    constexpr real_t PI = 3.14159265358979323846f;
    constexpr real_t DEG_TO_RAD = PI / 180.0f;
    constexpr real_t RAD_TO_DEG = 180.0f / PI;
    constexpr real_t EPSILON = 1e-6f;

    struct Vector2 {
        real_t a;
        real_t b;

        constexpr Vector2(real_t a_ = 0.0f, real_t b_ = 0.0f)
             : a(a_), b(b_) {}
    };

    struct Bounds2D {
        real_t min_x;
        real_t max_x;
        real_t min_y;
        real_t max_y;

        constexpr Bounds2D(real_t min_x_, real_t max_x_, real_t min_y_, real_t max_y_)
            : min_x(min_x_), max_x(max_x_), min_y(min_y_), max_y(max_y_) {}

        bool is_outside(real_t x, real_t y) const {
            return x < min_x || x > max_x || y < min_y || y > max_y;
        }

        real_t random_x(std::mt19937& gen) const {
            std::uniform_real_distribution<real_t> dist(min_x, max_x);
            return dist(gen);
        }

        real_t random_y(std::mt19937& gen) const {
            std::uniform_real_distribution<real_t> dist(min_y, max_y);
            return dist(gen);
        }
    };

    //--------------------------------------------------------------------------------
    // Display Settings
    //--------------------------------------------------------------------------------
    namespace Display
    {
        constexpr index_type WIDTH = 1000;    // 100m in simulation
        constexpr index_type HEIGHT = 1000;   // 100m in simulation
        constexpr index_type FPS = 30;

        // Color definitions (RGB format for SDL2)
        const color_rgb BLACK   = {0, 0, 0};
        const color_rgb WHITE   = {255, 255, 255};
        const color_rgb RED     = {255, 0, 0};
        const color_rgb GREEN   = {0, 255, 0};
        const color_rgb BLUE    = {0, 0, 255};
        const color_rgb ORANGE  = {255, 102, 0};

        static SDL_Color to_sdl_color(const color_rgb& color, uint8_t alpha = 255) {
            return SDL_Color{color[0], color[1], color[2], alpha};
        }
    }

    //--------------------------------------------------------------------------------
    // Section Line Settings
    //--------------------------------------------------------------------------------
    namespace Section
    {
        constexpr real_t GOAL_LINE = static_cast<real_t>(Display::HEIGHT) * 0.1f;   // Goal 구역 끝점
        constexpr real_t START_LINE = static_cast<real_t>(Display::HEIGHT) * 0.9f;  // Agent 구역 시작점
    }

    //--------------------------------------------------------------------------------
    // Goal Settings
    //--------------------------------------------------------------------------------
    namespace Goal
    {
        constexpr real_t RADIUS = 10.0f;
        constexpr real_t WIDTH = 3.0f;

        const Bounds2D BOUNDS(
            RADIUS + WIDTH,
            static_cast<real_t>(Display::WIDTH) - (RADIUS + WIDTH),
            RADIUS + WIDTH,
            Section::GOAL_LINE - (RADIUS + WIDTH)
        );
    }

    //--------------------------------------------------------------------------------
    // Agent Settings
    //--------------------------------------------------------------------------------
    namespace Agent
    {
        constexpr real_t  RADIUS = 10.0f;

        // Movement boundaries
        const Bounds2D BOUNDS(
            RADIUS,
            static_cast<real_t>(Display::WIDTH) - RADIUS,
            Section::START_LINE + RADIUS,
            static_cast<real_t>(Display::HEIGHT) - RADIUS
        );

        // Movement constraints
        const Vector2 VELOCITY_LIMITS(30.0f, 50.0f);  // Range: 3~5 m/s
        constexpr real_t YAW_RATE_LIMIT = 51.5f * DEG_TO_RAD;  // 51.5 degrees/s

        // Field of View (FOV) properties
        namespace FOV
        {
            constexpr real_t ANGLE = 103.0f * DEG_TO_RAD;  // 103 degrees
            constexpr real_t RANGE = 500.0f;  // 50m in simulation
            constexpr index_type RAY_COUNT = 150;
        }
    }

    //--------------------------------------------------------------------------------
    // Obstacle Settings
    //--------------------------------------------------------------------------------
    namespace Obstacle
    {
        constexpr index_type COUNT = 50;
        constexpr real_t RADIUS = 10.0f;

        // Movement boundaries
        const Bounds2D BOUNDS(
            RADIUS,
            static_cast<real_t>(Display::WIDTH) - RADIUS,
            Section::GOAL_LINE + RADIUS,
            Section::START_LINE - RADIUS
        );

        // Movement constraints
        const Vector2 VELOCITY_LIMITS(10.0f, 15.0f);  // Range: 1~1.5 m/s
    }
}
