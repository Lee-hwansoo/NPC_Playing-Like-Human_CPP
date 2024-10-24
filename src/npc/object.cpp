#include "npc/object.hpp"

namespace object {

CircleObstacle::CircleObstacle(std::optional<float> x, std::optional<float> y,
                             float radius, const Boundary& limit,
                             const SDL_Color& color, bool type)
    : Object(x.value_or(0.0f), y.value_or(0.0f), limit, color, type)
    , radius_(radius)
    , velocity_(0.0f)
    , yaw_(0.0f)
    , yaw_rate_(0.0f) {

    reset(x, y);
}

void CircleObstacle::reset(std::optional<float> x, std::optional<float> y) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (x.has_value() && y.has_value()) {
        position_ = torch::tensor({x.value(), y.value()}, torch::kFloat32);
    } else {
        position_ = torch::tensor({limit_.random_x(gen), limit_.random_y(gen)}, torch::kFloat32);
    }

    if (type_) {
        std::uniform_real_distribution<float> dist_vel(0.0f, 100.0f);
        std::uniform_real_distribution<float> dist_yaw(-M_PI, M_PI);
        std::uniform_real_distribution<float> dist_yaw_rate(-M_PI/6, M_PI/6);

        velocity_ = dist_vel(gen);
        yaw_ = dist_yaw(gen);
        yaw_rate_ = dist_yaw_rate(gen);
    } else {
        velocity_ = 0.0f;
        yaw_ = 0.0f;
        yaw_rate_ = 0.0f;
    }
}

void CircleObstacle::update(float dt) {
    if (!type_) return;

    yaw_ += yaw_rate_ * dt;
    yaw_ = std::fmod(yaw_ + M_PI, 2 * M_PI) - M_PI;

    auto movement = torch::tensor({std::cos(yaw_), std::sin(yaw_)}, torch::kFloat32);
    auto new_position = position_ + velocity_ * movement * dt;

    if (check_collision(new_position)) {
        yaw_ = std::fmod(yaw_ + M_PI, 2 * M_PI);
        add_random_movement();
        movement = torch::tensor({std::cos(yaw_), std::sin(yaw_)}, torch::kFloat32);
        new_position = position_ + velocity_ * movement * dt;
    }

    position_ = new_position;
}

bool CircleObstacle::check_collision(const torch::Tensor& new_position) {
    return limit_.is_outside(new_position[0].item<float>(), new_position[1].item<float>());
}

void CircleObstacle::add_random_movement() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-M_PI/6, M_PI/6);
    yaw_rate_ = dist(gen);
}

void CircleObstacle::draw(SDL_Renderer* renderer) {
    for (int w = 0; w < radius_ * 2; w++) {
        for (int h = 0; h < radius_ * 2; h++) {
            int dx = radius_ - w;
            int dy = radius_ - h;
            if ((dx*dx + dy*dy) <= (radius_*radius_)) {
                SDL_SetRenderDrawColor(renderer, color_.r, color_.g, color_.b, color_.a);
                SDL_RenderDrawPoint(renderer,
                                  position_[0].item<float>() + dx,
                                  position_[1].item<float>() + dy);
            }
        }
    }
}

torch::Tensor CircleObstacle::get_state() const {
    auto state = torch::cat({position_, torch::tensor({radius_}, torch::kFloat32)});
    return state;
}

Goal::Goal(std::optional<float> x, std::optional<float> y,
         float radius, const Boundary& limit,
         const SDL_Color& color, bool type)
    : Object(x.value_or(0.0f), y.value_or(0.0f), limit, color, type)
    , radius_(radius) {

    reset(x, y);
}

void Goal::reset(std::optional<float> x, std::optional<float> y) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (x.has_value() && y.has_value()) {
        position_ = torch::tensor({x.value(), y.value()}, torch::kFloat32);
    } else {
        position_ = torch::tensor({limit_.random_x(gen), limit_.random_y(gen)}, torch::kFloat32);
    }
}

void Goal::draw(SDL_Renderer* renderer) {
    const int32_t diameter = (radius_ * 2);
    const int32_t x = position_[0].item<float>() - radius_;
    const int32_t y = position_[1].item<float>() - radius_;

    SDL_SetRenderDrawColor(renderer, color_.r, color_.g, color_.b, color_.a);

    for (int w = 0; w < diameter; w++) {
        for (int h = 0; h < diameter; h++) {
            int dx = radius_ - w;
            int dy = radius_ - h;
            int dist = dx*dx + dy*dy;
            if (dist <= (radius_*radius_) &&
                dist >= ((radius_- 3)*(radius_- 3))) {
                SDL_RenderDrawPoint(renderer, x + w, y + h);
            }
        }
    }
}

torch::Tensor Goal::get_state() const {
    auto state = torch::cat({position_, torch::tensor({radius_}, torch::kFloat32)});
    return state;
}

} // namespace object
