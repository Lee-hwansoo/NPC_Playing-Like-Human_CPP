#pragma once

#include "utils/types.hpp"
#include "utils/constants.hpp"
#include "npc/path_planning.hpp"
#include <SDL.h>
#include <memory>
#include <torch/torch.h>
#include <optional>

namespace object {

using namespace types;
using namespace constants;

class Object {
public:
    Object(real_t x, real_t y,
           const Bounds2D& spawn_limit,
           const SDL_Color& color,
           bool type)
        : position_(torch::tensor({x, y}, get_tensor_dtype()))
        , spawn_limit_(spawn_limit)
        , color_(color)
        , type_(type) {}

    virtual ~Object() = default;

    virtual void reset(std::optional<real_t> x = std::nullopt, std::optional<real_t> y = std::nullopt) {}
    virtual void update(real_t dt) {}
    virtual bool check_bounds(const tensor_t& new_position) {return spawn_limit_.is_outside(new_position[0].item<real_t>(), new_position[1].item<real_t>());}
    virtual void draw(SDL_Renderer* renderer) = 0;

    virtual tensor_t get_state() const { return position_; }
    bool is_dynamic() const { return type_; }

protected:
    tensor_t position_;
    Bounds2D spawn_limit_;
    SDL_Color color_;
    bool type_;  // True: Dynamic, False: Static
};

class CircleObstacle : public Object {
public:
    CircleObstacle(std::optional<real_t> x = std::nullopt,
                   std::optional<real_t> y = std::nullopt,
                   real_t radius = constants::CircleObstacle::RADIUS,
                   const Bounds2D& spawn_limit = constants::CircleObstacle::SPAWN_BOUNDS,
                   const SDL_Color& color = Display::to_sdl_color(Display::ORANGE),
                   bool type = true);

    void reset(std::optional<real_t> x = std::nullopt, std::optional<real_t> y = std::nullopt) override;
    void update(real_t dt) override;
    tensor_t get_state() const override;
    void draw(SDL_Renderer* renderer) override;

private:
    real_t radius_;
    real_t force_;
    real_t yaw_;
    real_t yaw_rate_;

    void add_random_movement();
};

class RectangleObstacle : public Object {
public:
    RectangleObstacle(std::optional<real_t> x = std::nullopt,
                   std::optional<real_t> y = std::nullopt,
                   std::optional<real_t> width = std::nullopt,
                   std::optional<real_t> height = std::nullopt,
                   std::optional<real_t> yaw = std::nullopt,
                   const Bounds2D& spawn_limit = constants::RectangleObstacle::SPAWN_BOUNDS,
                   const SDL_Color& color = Display::to_sdl_color(Display::ORANGE),
                   bool type = false);

    void reset(std::optional<real_t> x = std::nullopt, std::optional<real_t> y = std::nullopt, std::optional<real_t> width = std::nullopt, std::optional<real_t> height = std::nullopt, std::optional<real_t> yaw = std::nullopt);
    tensor_t get_state() const override;
    void draw(SDL_Renderer* renderer) override;

private:
    real_t width_;
    real_t height_;
    real_t yaw_;

    std::array<tensor_t, 4> get_corners() const;
};

class Goal : public Object {
public:
    Goal(std::optional<real_t> x = std::nullopt,
         std::optional<real_t> y = std::nullopt,
         real_t radius = constants::Goal::RADIUS,
         const Bounds2D& spawn_limit = constants::Goal::SPAWN_BOUNDS,
         const SDL_Color& color = Display::to_sdl_color(Display::GREEN),
         bool type = false);

    void reset(std::optional<real_t> x = std::nullopt, std::optional<real_t> y = std::nullopt) override;
    tensor_t get_state() const override;
    void draw(SDL_Renderer* renderer) override;

private:
    real_t radius_;
};

class Agent : public Object {
public:
    Agent(std::optional<real_t> x = std::nullopt,
          std::optional<real_t> y = std::nullopt,
          real_t radius = constants::Agent::RADIUS,
          const Bounds2D& spawn_limit = constants::Agent::SPAWN_BOUNDS,
          const Bounds2D& move_limit = constants::Agent::MOVE_BOUNDS,
          const SDL_Color& color = Display::to_sdl_color(Display::BLUE),
          bool type = true,
          const tensor_t& circle_obstacles_state = torch::tensor({}),
          const tensor_t& rectangle_obstacles_state = torch::tensor({}),
          const tensor_t& goal_state = torch::tensor({}));

    tensor_t reset(std::optional<real_t> x = std::nullopt, std::optional<real_t> y = std::nullopt, const tensor_t& circle_obstacles_state = torch::tensor({}), const tensor_t& rectangle_obstacles_state = torch::tensor({}), const tensor_t& goal_state = torch::tensor({}));
    tensor_t update(const real_t dt, const tensor_t& scaled_action, const tensor_t& circle_obstacles_state, const tensor_t& goal_state);
    tensor_t get_state() const override;
    bool check_bounds(const tensor_t& new_position) override { return move_limit_.is_outside(new_position[0].item<real_t>(), new_position[1].item<real_t>()); }
    void draw(SDL_Renderer* renderer) override;

private:
    real_t radius_;
    tensor_t velocity_;
    real_t yaw_;
    Bounds2D move_limit_;
    const tensor_t& circle_obstacles_state_;
    const tensor_t& rectangle_obstacles_state_;
    const tensor_t& goal_state_;

    tensor_t trajectory_;

    std::unique_ptr<path_planning::RRT> path_planner_;
    tensor_t initial_path_;
    Vector2 frenet_point_;
    real_t frenet_d_;

    tensor_t fov_points_;
    tensor_t fov_distances_;
    real_t goal_distance_;
    real_t angle_to_goal_;
    bool is_goal_in_fov_;

    std::tuple<tensor_t, tensor_t, real_t, real_t, bool> calculate_fov(
        const tensor_t& agent_pos,
        const real_t& agent_angle,
        const tensor_t& circle_obstacles_state,
        const tensor_t& rectangle_obstacles_state,
        const tensor_t& goal_state
    );

    index_type get_closest_waypoint();
    std::tuple<Vector2, real_t> get_frenet_d();

};

} // namespace object
