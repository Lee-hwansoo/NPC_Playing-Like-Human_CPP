#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include <SDL.h>
#include <torch/torch.h>
#include <optional>
#include <random>
#include <cmath>

namespace object {

using namespace types;
using namespace constants;

class Object {
public:
    Object(real_t x, real_t y,
           const Bounds2D& limit,
           const SDL_Color& color,
           bool type)
        : position_(torch::tensor({x, y}, get_tensor_dtype()))
        , limit_(limit)
        , color_(color)
        , type_(type) {}

    virtual ~Object() = default;

    virtual void reset(std::optional<real_t> x = std::nullopt, std::optional<real_t> y = std::nullopt) = 0;
    virtual void update(real_t dt) {}
    virtual bool check_collision(const tensor_t& new_position) {return false;}
    virtual void draw(SDL_Renderer* renderer) = 0;

    virtual tensor_t get_state() const { return position_; }
    bool is_dynamic() const { return type_; }

protected:
    tensor_t position_;
    Bounds2D limit_;
    SDL_Color color_;
    bool type_;  // True: Dynamic, False: Static
};

class CircleObstacle : public Object {
public:
    CircleObstacle(std::optional<real_t> x = std::nullopt,
                   std::optional<real_t> y = std::nullopt,
                   real_t radius = constants::Obstacle::RADIUS,
                   const Bounds2D& limit = constants::Obstacle::BOUNDS,
                   const SDL_Color& color = Display::to_sdl_color(Display::ORANGE),
                   bool type = true);

    void reset(std::optional<real_t> x = std::nullopt, std::optional<real_t> y = std::nullopt) override;
    void update(real_t dt) override;
    bool check_collision(const tensor_t& new_position) override;
    void draw(SDL_Renderer* renderer) override;
    tensor_t get_state() const override;

private:
    void add_random_movement();

    real_t radius_;
    real_t velocity_;
    real_t yaw_;
    real_t yaw_rate_;
};

class Goal : public Object {
public:
    Goal(std::optional<real_t> x = std::nullopt,
         std::optional<real_t> y = std::nullopt,
         real_t radius = constants::Goal::RADIUS,
         const Bounds2D& limit = constants::Goal::BOUNDS,
         const SDL_Color& color = Display::to_sdl_color(Display::GREEN),
         bool type = false);

    void reset(std::optional<real_t> x = std::nullopt, std::optional<real_t> y = std::nullopt) override;
    void draw(SDL_Renderer* renderer) override;
    tensor_t get_state() const override;

private:
    real_t radius_;
};

} // namespace object
