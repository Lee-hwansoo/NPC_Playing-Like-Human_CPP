#pragma once

#include <SDL.h>
#include <torch/torch.h>
#include <optional>
#include <random>
#include <cmath>

namespace object {

struct Boundary {
    float x_min;
    float x_max;
    float y_min;
    float y_max;

    bool is_outside(float x, float y) const {
        return x < x_min || x > x_max || y < y_min || y > y_max;
    }

    float random_x(std::mt19937& gen) const {
        std::uniform_real_distribution<float> dist(x_min, x_max);
        return dist(gen);
    }

    float random_y(std::mt19937& gen) const {
        std::uniform_real_distribution<float> dist(y_min, y_max);
        return dist(gen);
    }
};

class Object {
public:
    Object(float x, float y,
           const Boundary& limit,
           const SDL_Color& color,
           bool type)
        : position_(torch::tensor({x, y}, torch::kFloat32))
        , limit_(limit)
        , color_(color)
        , type_(type) {}

    virtual ~Object() = default;

    virtual void reset(std::optional<float> x = std::nullopt, std::optional<float> y = std::nullopt) = 0;
    virtual void update(float dt) {}
    virtual bool check_collision(const torch::Tensor& new_position) {return false;}
    virtual void draw(SDL_Renderer* renderer) = 0;

    virtual torch::Tensor get_state() const { return position_; }
    bool is_dynamic() const { return type_; }

protected:
    torch::Tensor position_;
    Boundary limit_;
    SDL_Color color_;
    bool type_;  // True: Dynamic, False: Static
};

class CircleObstacle : public Object {
public:
    CircleObstacle(std::optional<float> x = std::nullopt,
                   std::optional<float> y = std::nullopt,
                   float radius = 15.0f,
                   const Boundary& limit = {0.f, 800.f, 0.f, 600.f},
                   const SDL_Color& color = {0, 0, 0, 255},
                   bool type = true);

    void reset(std::optional<float> x = std::nullopt, std::optional<float> y = std::nullopt) override;
    void update(float dt) override;
    bool check_collision(const torch::Tensor& new_position) override;
    void draw(SDL_Renderer* renderer) override;
    torch::Tensor get_state() const override;

private:
    void add_random_movement();

    float radius_;
    float velocity_;
    float yaw_;
    float yaw_rate_;
};

class Goal : public Object {
public:
    Goal(std::optional<float> x = std::nullopt,
         std::optional<float> y = std::nullopt,
         float radius = 20.0f,
         const Boundary& limit = {0.f, 800.f, 0.f, 600.f},
         const SDL_Color& color = {0, 255, 0, 255},
         bool type = false);

    void reset(std::optional<float> x = std::nullopt, std::optional<float> y = std::nullopt) override;
    void draw(SDL_Renderer* renderer) override;
    torch::Tensor get_state() const override;

private:
    float radius_;
};

} // namespace object
