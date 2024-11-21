#pragma once

#include "utils/types.hpp"
#include <SDL.h>

namespace constants
{
    using namespace types;

    constexpr real_t PI = 3.14159265358979323846f;
    constexpr real_t DEG_TO_RAD = PI / 180.0f;
    constexpr real_t RAD_TO_DEG = 180.0f / PI;
    constexpr real_t EPSILON = 1e-6f;

    constexpr const char* LOG_DIR = "../../logs";
    constexpr const char* HIS_DIR = "../../results";

    //--------------------------------------------------------------------------------
    // Display Settings
    //--------------------------------------------------------------------------------
    namespace Display
    {
        constexpr const char* WINDOW_TITLE = "Pearl Abyss";

        constexpr count_type WIDTH = 1000;    // 100m in simulation
        constexpr count_type HEIGHT = 1000;   // 100m in simulation
        constexpr count_type FPS = 30;

        // Color definitions (RGB format for SDL2)
        const color_rgb BLACK   = {0, 0, 0};
        const color_rgb GRAY    = {100, 100, 100};
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

        const Bounds2D SPAWN_BOUNDS(
            RADIUS + WIDTH,
            static_cast<real_t>(Display::WIDTH) - (RADIUS + WIDTH),
            (RADIUS + WIDTH),
            Section::GOAL_LINE - (RADIUS + WIDTH)
        );
    }

    //--------------------------------------------------------------------------------
    // Agent Settings
    //--------------------------------------------------------------------------------
    namespace Agent
    {
        constexpr real_t RADIUS = 10.0f;

        // Movement boundaries
        const Bounds2D SPAWN_BOUNDS(
            RADIUS,
            static_cast<real_t>(Display::WIDTH) - RADIUS,
            Section::START_LINE + RADIUS,
            static_cast<real_t>(Display::HEIGHT) - RADIUS
        );

		const Bounds2D MOVE_BOUNDS(
			RADIUS,
			static_cast<real_t>(Display::WIDTH) - RADIUS,
			RADIUS,
			static_cast<real_t>(Display::HEIGHT) - RADIUS
		);

        // Movement constraints
        const Vector2 VELOCITY_LIMITS(0.0f, 50.0f);  // Range: 0~5 m/s
        constexpr real_t YAW_CHANGE_LIMIT = 120.0f * DEG_TO_RAD;  // 120 degrees/s

        // Field of View (FOV) properties
        namespace FOV
        {
            constexpr real_t ANGLE = 280.0f * DEG_TO_RAD;  // 280 degrees
            constexpr real_t RANGE = 200.0f;  // 20m in simulation
            constexpr count_type RAY_COUNT = 56;
        }
    }

    //--------------------------------------------------------------------------------
    // Obstacle Settings
    //--------------------------------------------------------------------------------
    namespace CircleObstacle
    {
        constexpr count_type COUNT = 0;
        constexpr real_t RADIUS = 10.0f;

        // Movement boundaries
        const Bounds2D SPAWN_BOUNDS(
            RADIUS,
            static_cast<real_t>(Display::WIDTH) - RADIUS,
            Section::GOAL_LINE + RADIUS,
            Section::START_LINE - RADIUS
        );

        // Movement constraints
        const Vector2 VELOCITY_LIMITS(8.0f, 8.0f);  // Range: 0.8~0.8 m/s
    }

    namespace RectangleObstacle
    {
        constexpr count_type COUNT = 20;

        const Vector2 WIDTH_LIMITS(5.0f, 80.0f);   // Range: 0.5~8m
        const Vector2 HEIGHT_LIMITS(5.0f, 80.0f);  // Range: 0.5~8m

        // Movement boundaries
        const Bounds2D SPAWN_BOUNDS(
            WIDTH_LIMITS.a,
            static_cast<real_t>(Display::WIDTH) - WIDTH_LIMITS.a,
            Section::GOAL_LINE + HEIGHT_LIMITS.b,
            Section::START_LINE - HEIGHT_LIMITS.b
        );
    }

    //--------------------------------------------------------------------------------
    // PathPlanning RRT Settings
    //--------------------------------------------------------------------------------
    namespace RRT
    {
        constexpr count_type MAX_ATTEMPTS = 2;
        constexpr count_type MAX_ITER = 2500;
        constexpr real_t GOAL_SAMPLE_RATE = 0.1f;
        constexpr real_t MIN_U = 25.0f;
        constexpr real_t MAX_U = 100.0f;
        constexpr real_t SUCCESS_DIST_THRESHOLD = Goal::RADIUS - Goal::WIDTH;
        constexpr real_t COLLISION_CHECK_STEP = 0.2f;
        constexpr real_t STEP_SIZE = 0.35f;
    }

	//--------------------------------------------------------------------------------
	// SAC Settings
	//--------------------------------------------------------------------------------
    namespace NETWORK
    {
        constexpr real_t  MAX_STEP = 4000;
		constexpr count_type BUFFER_SIZE = 700000;
		constexpr count_type BATCH_SIZE = 256;
		constexpr real_t  GAMMA = 0.995f;
		constexpr real_t  TAU = 0.005f;
		constexpr real_t  ALPHA = 0.2f;
        constexpr real_t  LEARNING_RATE = 3e-4f;

        constexpr count_type UPDATE_INTERVAL = 5;
        constexpr count_type LOG_INTERVAL = 25;

        constexpr types::count_type N_STEPS = 4;
    }
}
