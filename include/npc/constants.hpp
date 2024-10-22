#pragma once

#include <Eigen/Dense>
#include <array>

namespace Constants
{
    // Mathematical constants
    constexpr double PI = 3.14159265358979323846;
    constexpr double DEG_TO_RAD = PI / 180.0;

    // Type aliases for better readability
    using ColorRGB = std::array<int, 3>;
    using Vector2d = Eigen::Vector2d;
    using Matrix2d = Eigen::Matrix<double, 2, 2>;

    //--------------------------------------------------------------------------------
    // Display Settings
    //--------------------------------------------------------------------------------
    namespace Display
    {
        constexpr int WIDTH = 1000;    // 100m in simulation
        constexpr int HEIGHT = 1000;   // 100m in simulation
        constexpr int FPS = 30;

        // Color definitions (RGB format for SDL2)
        const ColorRGB BLACK   = {0, 0, 0};
        const ColorRGB WHITE   = {255, 255, 255};
        const ColorRGB RED     = {255, 0, 0};
        const ColorRGB GREEN   = {0, 255, 0};
        const ColorRGB BLUE    = {0, 0, 255};
        const ColorRGB ORANGE  = {255, 102, 0};
    }

    //--------------------------------------------------------------------------------
    // Goal Settings
    //--------------------------------------------------------------------------------
    namespace Goal
    {
        constexpr double RADIUS = 10.0;
        constexpr double WIDTH = 3.0;

        const Matrix2d BOUNDS = (Matrix2d() <<
            RADIUS + WIDTH, Display::WIDTH - (RADIUS + WIDTH),
            RADIUS + WIDTH, Display::HEIGHT * 0.1 - (RADIUS + WIDTH)
        ).finished();
    }

    //--------------------------------------------------------------------------------
    // Agent Settings
    //--------------------------------------------------------------------------------
    namespace Agent
    {
        constexpr double RADIUS = 10.0;

        // Movement boundaries
        const Matrix2d BOUNDS = (Matrix2d() <<
            RADIUS, Display::WIDTH - RADIUS,
            Display::HEIGHT * 0.9 + RADIUS, Display::HEIGHT - RADIUS
        ).finished();

        // Movement constraints
        const Vector2d VELOCITY_LIMITS(30.0, 50.0);  // Range: 3~5 m/s
        constexpr double YAW_RATE_LIMIT = 51.5 * DEG_TO_RAD;  // 51.5 degrees/s

        // Field of View (FOV) properties
        namespace FOV
        {
            constexpr double ANGLE = 103.0 * DEG_TO_RAD;  // 103 degrees
            constexpr double RANGE = 500.0;  // 50m in simulation
            constexpr int RAY_COUNT = 150;
        }
    }

    //--------------------------------------------------------------------------------
    // Obstacle Settings
    //--------------------------------------------------------------------------------
    namespace Obstacle
    {
        constexpr int COUNT = 50;
        constexpr double RADIUS = 10.0;

        // Movement boundaries
        const Matrix2d BOUNDS = (Matrix2d() <<
            RADIUS, Display::WIDTH - RADIUS,
            Goal::BOUNDS(1, 1) + (Goal::RADIUS + Goal::WIDTH) + RADIUS,
            Agent::BOUNDS(1, 0) + Agent::RADIUS - RADIUS
        ).finished();

        // Movement constraints
        const Vector2d VELOCITY_LIMITS(10.0, 15.0);  // Range: 1~1.5 m/s
    }
}
