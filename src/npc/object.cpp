#include "npc/object.hpp"
#include <memory>
#include <random>

namespace object {

CircleObstacle::CircleObstacle(std::optional<real_t> x,
                            std::optional<real_t> y,
                            real_t radius,
                            const Bounds2D& spawn_limit,
                            const SDL_Color& color,
                            bool type)
    : Object(x.value_or(0.0f), y.value_or(0.0f), spawn_limit, color, type)
    , radius_(radius)
    , force_(0.0f)
    , yaw_(0.0f)
    , yaw_rate_(0.0f) {
    reset(x, y);
}

void CircleObstacle::reset(std::optional<real_t> x, std::optional<real_t> y) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (x.has_value() && y.has_value()) {
        position_ = torch::tensor({x.value(), y.value()});
    } else {
        position_ = torch::tensor({spawn_limit_.random_x(gen), spawn_limit_.random_y(gen)});
    }

    if (type_) {
        std::uniform_real_distribution<real_t> dist_force(constants::CircleObstacle::VELOCITY_LIMITS.a, constants::CircleObstacle::VELOCITY_LIMITS.b);
        std::uniform_real_distribution<real_t> dist_yaw(-constants::PI, constants::PI);
        std::uniform_real_distribution<real_t> dist_yaw_rate(-constants::PI/6, constants::PI/6);

        force_ = dist_force(gen);
        yaw_ = dist_yaw(gen);
        yaw_rate_ = dist_yaw_rate(gen);
    } else {
        force_ = 0.0f;
        yaw_ = 0.0f;
        yaw_rate_ = 0.0f;
    }
}

void CircleObstacle::update(real_t dt) {
    if (!type_) return;

    yaw_ += yaw_rate_ * dt;
    yaw_ = std::fmod(yaw_ + constants::PI, 2 * constants::PI) - constants::PI;

    auto movement = torch::tensor({std::cos(yaw_), std::sin(yaw_)});
    auto new_position = position_ + force_ * movement * dt;

    if (this->check_bounds(new_position)) {
        yaw_ = std::fmod(yaw_ + constants::PI, 2 * constants::PI);
        add_random_movement();
        movement = torch::tensor({std::cos(yaw_), std::sin(yaw_)});
        new_position = position_ + force_ * movement * dt;
    }

    position_ = new_position;
}

void CircleObstacle::add_random_movement() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<real_t> dist(-constants::PI/6, constants::PI/6);
    yaw_rate_ = dist(gen);
}

tensor_t CircleObstacle::get_state() const {
    tensor_t state = torch::tensor({
			position_[0].item<real_t>(),
			position_[1].item<real_t>(),
			radius_
		});
    return state.unsqueeze(0);
}

void CircleObstacle::draw(SDL_Renderer* renderer) {
    index_type x = static_cast<index_type>(position_[0].item<real_t>());
    index_type y = static_cast<index_type>(position_[1].item<real_t>());
    index_type r = static_cast<index_type>(radius_);

    SDL_SetRenderDrawColor(renderer, color_.r, color_.g, color_.b, color_.a);

    // Bresenham's circle algorithm with filling
    index_type offsetx = 0;
    index_type offsety = r;
    index_type d = r - 1;

    while (offsety >= offsetx) {
        const SDL_Point points[] = {
            {x + offsetx, y + offsety}, {x + offsetx, y - offsety},
            {x - offsetx, y + offsety}, {x - offsetx, y - offsety},
            {x + offsety, y + offsetx}, {x + offsety, y - offsetx},
            {x - offsety, y + offsetx}, {x - offsety, y - offsetx}
        };
        SDL_RenderDrawPoints(renderer, points, 8);

        if (d >= 2*offsetx) {
            d -= 2*offsetx + 1;
            offsetx += 1;
        }
        else if (d < 2*(r-offsety)) {
            d += 2*offsety - 1;
            offsety -= 1;
        }
        else {
            d += 2*(offsety - offsetx - 1);
            offsety -= 1;
            offsetx += 1;
        }
    }
}

RectangleObstacle::RectangleObstacle(std::optional<real_t> x,
                                    std::optional<real_t> y,
                                    std::optional<real_t> width,
                                    std::optional<real_t> height,
                                    std::optional<real_t> yaw,
                                    const Bounds2D& spawn_limit,
                                    const SDL_Color& color,
                                    bool type)
    : Object(x.value_or(0.0f), y.value_or(0.0f), spawn_limit, color, type)
    , width_(width.value_or(0.0f))
    , height_(height.value_or(0.0f))
    , yaw_(yaw.value_or(0.0f)) {
    reset(x, y, width, height, yaw);
}

void RectangleObstacle::reset(std::optional<real_t> x, std::optional<real_t> y, std::optional<real_t> width, std::optional<real_t> height, std::optional<real_t> yaw) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (x.has_value() && y.has_value()) {
        position_ = torch::tensor({x.value(), y.value()});
    } else {
        position_ = torch::tensor({spawn_limit_.random_x(gen), spawn_limit_.random_y(gen)});
    }

    width_ = width.has_value() ? width.value() : std::uniform_real_distribution<real_t>(constants::RectangleObstacle::WIDTH_LIMITS.a, constants::RectangleObstacle::WIDTH_LIMITS.b)(gen);
    height_ = height.has_value() ? height.value() : std::uniform_real_distribution<real_t>(constants::RectangleObstacle::HEIGHT_LIMITS.a, constants::RectangleObstacle::HEIGHT_LIMITS.b)(gen);
    yaw_ = yaw.has_value() ? yaw.value() : std::uniform_real_distribution<real_t>(-constants::PI, constants::PI)(gen);
}

tensor_t RectangleObstacle::get_state() const {
    tensor_t state = torch::tensor({
            position_[0].item<real_t>(),  // 좌상단 x
            position_[1].item<real_t>(),  // 좌상단 y
            width_,                       // 너비
            height_,                      // 높이
            yaw_                          // 회전각
        });
    return state.unsqueeze(0);
}

std::array<tensor_t, 4> RectangleObstacle::get_corners() const {
    real_t cos_yaw = std::cos(yaw_);
    real_t sin_yaw = std::sin(yaw_);

    real_t x = position_[0].item<real_t>();
    real_t y = position_[1].item<real_t>();

    std::array<std::pair<real_t, real_t>, 4> corner_offsets = {
        std::make_pair(0.0f, 0.0f),                // Top-left (origin)
        std::make_pair(width_, 0.0f),              // Top-right
        std::make_pair(width_, height_),           // Bottom-right
        std::make_pair(0.0f, height_)              // Bottom-left
    };

    std::array<tensor_t, 4> corners;
    for (size_t i = 0; i < 4; ++i) {
        real_t dx = corner_offsets[i].first;
        real_t dy = corner_offsets[i].second;

        real_t rotated_x = x + (dx * cos_yaw - dy * sin_yaw);
        real_t rotated_y = y + (dx * sin_yaw + dy * cos_yaw);

        corners[i] = torch::tensor({rotated_x, rotated_y});
    }

    return corners;
}

void RectangleObstacle::draw(SDL_Renderer* renderer) {
    auto corners = get_corners();

    SDL_SetRenderDrawColor(renderer, color_.r, color_.g, color_.b, color_.a);

    std::array<SDL_Point, 5> points;
    for (size_t i = 0; i < 4; ++i) {
        points[i] = {
            static_cast<int>(corners[i][0].item<real_t>()),
            static_cast<int>(corners[i][1].item<real_t>())
        };
    }
    points[4] = points[0];
    SDL_RenderDrawLines(renderer, points.data(), 5);
}

Goal::Goal(std::optional<real_t> x,
        std::optional<real_t> y,
        real_t radius,
        const Bounds2D& spawn_limit,
        const SDL_Color& color,
        bool type)
    : Object(x.value_or(0.0f), y.value_or(0.0f), spawn_limit, color, type)
    , radius_(radius) {
    reset(x, y);
}

void Goal::reset(std::optional<real_t> x, std::optional<real_t> y) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (x.has_value() && y.has_value()) {
        position_ = torch::tensor({x.value(), y.value()});
    } else {
        position_ = torch::tensor({spawn_limit_.random_x(gen), spawn_limit_.random_y(gen)});
    }
}

tensor_t Goal::get_state() const {
    tensor_t state = torch::tensor({
            position_[0].item<real_t>(),
            position_[1].item<real_t>(),
            radius_
        });
    return state.unsqueeze(0);
}

void Goal::draw(SDL_Renderer* renderer) {
    index_type x = static_cast<index_type>(position_[0].item<real_t>());
    index_type y = static_cast<index_type>(position_[1].item<real_t>());
    index_type r = static_cast<index_type>(radius_);
    index_type w = static_cast<index_type>(constants::Goal::WIDTH);

    SDL_SetRenderDrawColor(renderer, color_.r, color_.g, color_.b, color_.a);

    // Bresenham's circle drawing algorithm
    for (index_type curr_r = r; curr_r > r - w; curr_r--) {
        index_type offsetx = 0;
        index_type offsety = curr_r;
        index_type d = curr_r - 1;

        while (offsety >= offsetx) {
            const SDL_Point points[] = {
                {x + offsetx, y + offsety}, {x + offsetx, y - offsety},
                {x - offsetx, y + offsety}, {x - offsetx, y - offsety},
                {x + offsety, y + offsetx}, {x + offsety, y - offsetx},
                {x - offsety, y + offsetx}, {x - offsety, y - offsetx}
            };
            SDL_RenderDrawPoints(renderer, points, 8);

            if (d >= 2*offsetx) {
                d -= 2*offsetx + 1;
                offsetx += 1;
            }
            else if (d < 2*(curr_r-offsety)) {
                d += 2*offsety - 1;
                offsety -= 1;
            }
            else {
                d += 2*(offsety - offsetx - 1);
                offsety -= 1;
                offsetx += 1;
            }
        }
    }
}

Agent::Agent(std::optional<real_t> x,
            std::optional<real_t> y,
            real_t radius,
            const Bounds2D& spawn_limit,
            const Bounds2D& move_limit,
            const SDL_Color& color,
            bool type,
            const tensor_t& circle_obstacles_state,
            const tensor_t& rectangle_obstacles_state,
            const tensor_t& goal_state)
    : Object(x.value_or(0.0f), y.value_or(0.0f), spawn_limit, color, type)
    , radius_(radius)
    , velocity_(torch::tensor({0.0f, 0.0f}))
    , yaw_(0.0f)
    , move_limit_(move_limit)
    , circle_obstacles_state_(circle_obstacles_state)
    , rectangle_obstacles_state_(rectangle_obstacles_state)
    , goal_state_(goal_state)
    , path_planner_(std::make_unique<path_planning::RRT>(position_, move_limit, circle_obstacles_state, rectangle_obstacles_state, goal_state)) {
    reset(x, y, circle_obstacles_state, rectangle_obstacles_state, goal_state);
}

std::tuple<tensor_t, tensor_t, real_t, real_t, bool, bool> Agent::calculate_fov(const tensor_t& agent_pos, const real_t& agent_angle, const tensor_t& circle_obstacles_state, const tensor_t& rectangle_obstacles_state, const tensor_t& goal_state) {
    bool is_collision = false;

    // FOV를 이루는 ray들의 각도 계산 [num_rays]
    auto ray_angles = torch::linspace(agent_angle - constants::Agent::FOV::ANGLE / 2, agent_angle + constants::Agent::FOV::ANGLE / 2, constants::Agent::FOV::RAY_COUNT);

    // ray들의 방향 벡터 계산 [num_rays, 2]
    auto ray_cos = torch::cos(ray_angles);
    auto ray_sin = torch::sin(ray_angles);
    auto ray_directions = torch::stack({ray_cos, ray_sin}, 1);

    // 목표와의 상대 위치, 거리, 각도 계산
    auto goal_vector = goal_state.slice(0, 0, 2) - agent_pos;       // [2]
    auto goal_distance = torch::norm(goal_vector).item<real_t>();
    auto goal_angle = torch::atan2(goal_vector[1], goal_vector[0]).item<real_t>();
    auto angle_diff = std::fmod(goal_angle - agent_angle + constants::PI, 2 * constants::PI) - constants::PI;
    auto abs_angle_diff = std::abs(angle_diff);

    // 화면 경계와의 ray casting 계산
    auto x_dirs = ray_directions.select(1, 0);                      // [num_rays]
	auto y_dirs = ray_directions.select(1, 1);                      // [num_rays]

    // 각 ray와 화면 경계의 교차점까지의 거리 계산 [4, num_rays]
	auto border_distances = torch::stack({
        (-agent_pos[0].item<real_t>()) / x_dirs,
        (constants::Display::WIDTH - agent_pos[0].item<real_t>()) / x_dirs,
        (-agent_pos[1].item<real_t>()) / y_dirs,
        (constants::Display::HEIGHT - agent_pos[1].item<real_t>()) / y_dirs
    });

    // 음수 거리(반대 방향)를 무한대로 설정
    border_distances = torch::where(border_distances <= 0, torch::full_like(border_distances, std::numeric_limits<real_t>::infinity()), border_distances);
    // ray별 가장 가까운 경계 거리 계산, FOV 최대 범위로 제한 [num_rays]
    auto closest_distances = torch::min(std::get<0>(torch::min(border_distances, 0)), torch::full({constants::Agent::FOV::RAY_COUNT}, constants::Agent::FOV::RANGE));

    if (circle_obstacles_state.size(0) > 0 && circle_obstacles_state.size(1) >= 3) {
        // 모든 원형 장애물의 상태값 추출
        auto obstacles_pos = circle_obstacles_state.slice(1, 0, 2);                     // [num_circles, 2]
        // 에이전트와 각 원형 장애물 간의 거리 벡터 계산
        auto diff = agent_pos.unsqueeze(0) - obstacles_pos;                             // [num_circles, 2]

		// 충돌 검사
		auto distances_squared = torch::sum(diff * diff, 1);                                  // [num_circles]
		auto radii = circle_obstacles_state.select(1, 2) + constants::Agent::RADIUS;         // [num_circles]
		auto radii_squared = radii.pow(2);                                                   // [num_circles]
		if (torch::any(distances_squared <= radii_squared).item<bool>()) {
			is_collision = true;
		}

        // 각 ray에 대한 벡터 확장
        auto oc = diff.unsqueeze(1).expand({-1, constants::Agent::FOV::RAY_COUNT, 2});  // [num_circles, num_rays, 2]

        // ray casting을 위한 2차 방정식 계수 계산
		auto a = torch::sum(ray_directions * ray_directions, 1);                             // [num_rays]
		auto b = 2 * torch::sum(oc * ray_directions.unsqueeze(0), 2);                       // [num_circles, num_rays]
		auto c = torch::sum(oc * oc, 2) - circle_obstacles_state.select(1, 2).unsqueeze(1).pow(2);  // [num_circles, num_rays]

		// 판별식 계산
		auto discriminant = b.pow(2) - 4 * a * c;                                           // [num_circles, num_rays]
		auto valid_intersections = discriminant >= 0;                                        // [num_circles, num_rays]

        if (torch::any(valid_intersections).item<bool>()) {
            auto safe_discriminant = torch::where(valid_intersections, discriminant, torch::zeros_like(discriminant));

            // 교차점까지의 거리 계산 (더 가까운 교차점 선택)
            auto t = torch::where(valid_intersections, (-b - torch::sqrt(safe_discriminant)) / (2 * a), torch::full_like(b, std::numeric_limits<real_t>::infinity()));      // [num_circles, num_rays]
            t = torch::where(t < 0, torch::full_like(t, std::numeric_limits<real_t>::infinity()), t);

            // 모든 원형 장애물에 대해 가장 가까운 거리 선택
            auto [circle_obstacles_distances, _]  = torch::min(t, 0);                       // [num_rays]
            closest_distances = torch::min(closest_distances, circle_obstacles_distances);
        }
    }

	if (rectangle_obstacles_state.size(0) > 0 && rectangle_obstacles_state.size(1) >= 5) {
		// 모든 직사각형과 모든 레이에 대한 계산을 한번에 수행
		auto rect_positions = rectangle_obstacles_state.slice(1, 0, 2);                    // [num_rect, 2]
		auto rect_sizes = rectangle_obstacles_state.slice(1, 2, 4);                        // [num_rect, 2]
		auto rect_angles = rectangle_obstacles_state.select(1, 4);                         // [num_rect]

		// 회전 행렬 생성 [num_rect, 2, 2]
		auto cos_theta = torch::cos(rect_angles);
		auto sin_theta = torch::sin(rect_angles);
		auto rotation_matrices = torch::stack({
			torch::stack({cos_theta, sin_theta}, 1),
			torch::stack({-sin_theta, cos_theta}, 1)
			}, 1);                                                                            // [num_rect, 2, 2]

		// 에이전트 위치를 각 직사각형의 로컬 좌표계로 변환 [num_rect, 2]
		auto to_rect = agent_pos.unsqueeze(0) - rect_positions;                          // [num_rect, 2]
		auto local_agent_pos = torch::matmul(rotation_matrices, to_rect.unsqueeze(2)).squeeze(2);  // [num_rect, 2]

        // 충돌 검사 - 에이전트의 반경을 고려한 AABB 테스트
		auto x_collision = (local_agent_pos.select(1, 0) >= -constants::Agent::RADIUS) &
			(local_agent_pos.select(1, 0) <= rect_sizes.select(1, 0) + constants::Agent::RADIUS);
		auto y_collision = (local_agent_pos.select(1, 1) >= -constants::Agent::RADIUS) &
			(local_agent_pos.select(1, 1) <= rect_sizes.select(1, 1) + constants::Agent::RADIUS);

		if (torch::any(x_collision & y_collision).item<bool>()) {
			is_collision = true;
		}

		// 모든 레이 방향을 각 직사각형의 로컬 좌표계로 변환 [num_rect, num_rays, 2]
		auto local_ray_dirs = torch::matmul(
			rotation_matrices,                                                            // [num_rect, 2, 2]
			ray_directions.transpose(0, 1).unsqueeze(0).expand({ rectangle_obstacles_state.size(0), -1, -1 })  // [num_rect, 2, num_rays]
		);                                                                               // [num_rect, 2, num_rays]
		local_ray_dirs = local_ray_dirs.transpose(1, 2);                                // [num_rect, num_rays, 2]

		// 각 직사각형의 경계 계산
		auto x_min = torch::zeros_like(rect_sizes.select(1, 0));                        // [num_rect]
		auto x_max = rect_sizes.select(1, 0);                                           // [num_rect]
		auto y_min = torch::zeros_like(rect_sizes.select(1, 1));                        // [num_rect]
		auto y_max = rect_sizes.select(1, 1);                                           // [num_rect]

		// x 방향 경계와의 교차점 계산
		auto dir_x = local_ray_dirs.select(2, 0);                                      // [num_rect, num_rays]
		auto dir_y = local_ray_dirs.select(2, 1);                                      // [num_rect, num_rays]
		auto origin_x = local_agent_pos.select(1, 0).unsqueeze(1);                     // [num_rect, 1]
		auto origin_y = local_agent_pos.select(1, 1).unsqueeze(1);                     // [num_rect, 1]

		auto tx_min = (x_min.unsqueeze(1) - origin_x) / dir_x;                        // [num_rect, num_rays]
		auto tx_max = (x_max.unsqueeze(1) - origin_x) / dir_x;                        // [num_rect, num_rays]
		auto ty_min = (y_min.unsqueeze(1) - origin_y) / dir_y;                        // [num_rect, num_rays]
		auto ty_max = (y_max.unsqueeze(1) - origin_y) / dir_y;                        // [num_rect, num_rays]

		// x 경계 교차점에서의 y 좌표
		auto y_at_tx_min = origin_y + tx_min * dir_y;                                 // [num_rect, num_rays]
		auto y_at_tx_max = origin_y + tx_max * dir_y;                                 // [num_rect, num_rays]

		// y 경계 교차점에서의 x 좌표
		auto x_at_ty_min = origin_x + ty_min * dir_x;                                 // [num_rect, num_rays]
		auto x_at_ty_max = origin_x + ty_max * dir_x;                                 // [num_rect, num_rays]

		// 유효한 교차점 찾기
		auto valid_tx_min = (tx_min > 0) & (y_at_tx_min >= y_min.unsqueeze(1)) & (y_at_tx_min <= y_max.unsqueeze(1));
		auto valid_tx_max = (tx_max > 0) & (y_at_tx_max >= y_min.unsqueeze(1)) & (y_at_tx_max <= y_max.unsqueeze(1));
		auto valid_ty_min = (ty_min > 0) & (x_at_ty_min >= x_min.unsqueeze(1)) & (x_at_ty_min <= x_max.unsqueeze(1));
		auto valid_ty_max = (ty_max > 0) & (x_at_ty_max >= x_min.unsqueeze(1)) & (x_at_ty_max <= x_max.unsqueeze(1));

		// 무효한 교차점을 무한대로 설정
		tx_min = torch::where(valid_tx_min, tx_min, torch::full_like(tx_min, std::numeric_limits<real_t>::infinity()));
		tx_max = torch::where(valid_tx_max, tx_max, torch::full_like(tx_max, std::numeric_limits<real_t>::infinity()));
		ty_min = torch::where(valid_ty_min, ty_min, torch::full_like(ty_min, std::numeric_limits<real_t>::infinity()));
		ty_max = torch::where(valid_ty_max, ty_max, torch::full_like(ty_max, std::numeric_limits<real_t>::infinity()));

		// 모든 교차점 중 가장 가까운 거리 찾기
		auto all_distances = torch::stack({ tx_min, tx_max, ty_min, ty_max }, 0);       // [4, num_rect, num_rays]
		auto [min_rect_distances, _] = torch::min(all_distances, 0);                   // [num_rect, num_rays]
		auto [final_rect_distances, __] = torch::min(min_rect_distances, 0);           // [num_rays]

		closest_distances = torch::min(closest_distances, final_rect_distances);
	}

    bool goal_in_fov = false;
    if (goal_distance <= constants::Agent::FOV::RANGE && abs_angle_diff <= constants::Agent::FOV::ANGLE / 2) {
        index_type goal_ray_index = static_cast<index_type>((abs_angle_diff + constants::Agent::FOV::ANGLE / 2) / constants::Agent::FOV::ANGLE * (constants::Agent::FOV::RAY_COUNT - 1));
        goal_in_fov = (goal_distance - goal_state[2].item<real_t>() <= closest_distances[goal_ray_index].item<real_t>());
    }

    auto points = agent_pos.unsqueeze(0) + ray_directions * closest_distances.unsqueeze(1);
    auto fov_points = torch::cat({agent_pos.unsqueeze(0), points});

    return std::make_tuple(fov_points, closest_distances, goal_distance, angle_diff, goal_in_fov, is_collision);
}

index_type Agent::get_closest_waypoint() {
	tensor_t expanded_position = position_.unsqueeze(0).expand_as(initial_path_);
    tensor_t distances = torch::norm(initial_path_ - expanded_position, 2, 1);

	torch::Tensor min_idx;
	std::tie(std::ignore, min_idx) = torch::min(distances, 0);

	return min_idx.item<index_type>();
}

std::tuple<Vector2, real_t> Agent::get_frenet_d() {
    if (initial_path_.size(0) > 0) {
        index_type closest_waypoint = get_closest_waypoint();
        index_type next_waypoint = (closest_waypoint + 1) % initial_path_.size(0);

        tensor_t n_vec = initial_path_[next_waypoint] - initial_path_[closest_waypoint];
        tensor_t x_vec = position_ - initial_path_[next_waypoint];

        tensor_t dot_product = torch::dot(x_vec, n_vec);
        tensor_t n_vec_norm_squared = torch::dot(n_vec, n_vec);
        tensor_t proj_norm = dot_product / n_vec_norm_squared;
        tensor_t proj_vec = proj_norm * n_vec;

		tensor_t diff_vec = x_vec - proj_vec;
		real_t frenet_d = torch::norm(diff_vec).item<real_t>();

        tensor_t x_vec_3d = torch::zeros({ 3 });
        tensor_t n_vec_3d = torch::zeros({ 3 });

		x_vec_3d.index_put_({ torch::indexing::Slice(0, 2) }, x_vec);
		n_vec_3d.index_put_({ torch::indexing::Slice(0, 2) }, n_vec);

        tensor_t cross_product = torch::cross(x_vec_3d, n_vec_3d, 0);

		if (cross_product[2].item<double>() > 0) {
			frenet_d = -frenet_d;
		}

        real_t x = initial_path_[closest_waypoint][0].item<real_t>();
        real_t y = initial_path_[closest_waypoint][1].item<real_t>();
        return std::make_tuple(Vector2(x, y), frenet_d);
    }
    else {
        return std::make_tuple(Vector2(), 0.0f);
    }
}

tensor_t Agent::reset(std::optional<real_t> x, std::optional<real_t> y, const tensor_t& circle_obstacles_state, const tensor_t& rectangle_obstacles_state, const tensor_t& goal_state) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (x.has_value() && y.has_value()) {
        position_ = torch::tensor({x.value(), y.value()});
    } else {
        position_ = torch::tensor({spawn_limit_.random_x(gen), spawn_limit_.random_y(gen)});
    }

    velocity_ = torch::tensor({0.0f, 0.0f});
    yaw_ = -0.5f * constants::PI;
    trajectory_ = torch::stack({position_});;

    std::tie(fov_points_, fov_distances_, goal_distance_, angle_to_goal_, is_goal_in_fov_, is_collison_) = calculate_fov(position_, yaw_, circle_obstacles_state, rectangle_obstacles_state, goal_state);

	path_planner_->update(position_, circle_obstacles_state, rectangle_obstacles_state, goal_state);
	initial_path_ = path_planner_->plan();
	std::tie(frenet_point_, frenet_d_) = get_frenet_d();

    return get_state();
}

tensor_t Agent::update(const real_t dt, const tensor_t& scaled_action, const tensor_t& circle_obstacles_state, const tensor_t& goal_state){
    real_t force = scaled_action[0].item<real_t>() * constants::Agent::VELOCITY_LIMITS.b;
    real_t yaw_change = scaled_action[1].item<real_t>() * constants::Agent::YAW_CHANGE_LIMIT;

    yaw_ += std::clamp(yaw_change, -constants::Agent::YAW_CHANGE_LIMIT, constants::Agent::YAW_CHANGE_LIMIT) * dt;
    yaw_ = std::fmod(yaw_ + constants::PI, 2 * constants::PI) - constants::PI;

    force = std::clamp(force, 1.0f, 50.0f);
    velocity_ = force * torch::tensor({std::cos(yaw_), std::sin(yaw_)}) * dt;

    position_ = position_ + velocity_;
    yaw_ = std::atan2(velocity_[1].item<real_t>(), velocity_[0].item<real_t>());
    trajectory_ = torch::cat({trajectory_, position_.unsqueeze(0)});;

    std::tie(fov_points_, fov_distances_, goal_distance_, angle_to_goal_, is_goal_in_fov_, is_collison_) = calculate_fov(position_, yaw_, circle_obstacles_state, rectangle_obstacles_state_, goal_state);

    std::tie(frenet_point_, frenet_d_) = get_frenet_d();
    return get_state();
}

tensor_t Agent::get_state() const {
    auto normalized_position = position_ / torch::tensor({static_cast<real_t>(constants::Display::WIDTH), static_cast<real_t>(constants::Display::HEIGHT)});
    auto normalized_yaw = torch::tensor({yaw_ / constants::PI});
    auto normalized_velocity = velocity_ / constants::Agent::VELOCITY_LIMITS.b;
    auto normalized_fov_dist = fov_distances_.flatten() / constants::Agent::FOV::RANGE;
    auto normalized_goal_dist = torch::tensor({goal_distance_ / std::sqrt(constants::Display::WIDTH * constants::Display::WIDTH + constants::Display::HEIGHT * constants::Display::HEIGHT)});
    auto normalized_angle_diff = torch::tensor({angle_to_goal_ / constants::PI});
    auto goal_in_fov_tensor = torch::tensor({static_cast<real_t>(is_goal_in_fov_)});
    auto normalized_frenet_d = torch::tensor({frenet_d_ / (constants::Display::WIDTH > constants::Display::HEIGHT ? constants::Display::WIDTH : constants::Display::HEIGHT)});

    auto state = torch::cat({
        normalized_position,
        normalized_yaw,
        normalized_velocity,
        normalized_fov_dist,
        normalized_goal_dist,
        normalized_angle_diff,
        goal_in_fov_tensor,
        normalized_frenet_d
    });

    return state.unsqueeze(0);
};

tensor_t Agent::get_raw_state() const {
    tensor_t state = torch::tensor({
		position_[0].item<real_t>(),
		position_[1].item<real_t>(),
		radius_
		});
	return state.unsqueeze(0);
}

void Agent::draw(SDL_Renderer* renderer) {
    SDL_Color color = constants::Display::to_sdl_color(constants::Display::GRAY);
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 80);
    const auto& node_list = path_planner_->get_node_list();
    for (const auto& node : node_list) {
        if (node->parent()) {
            SDL_RenderDrawLine(renderer,
                static_cast<index_type>(node->x()),
                static_cast<index_type>(node->y()),
                static_cast<index_type>(node->parent()->x()),
                static_cast<index_type>(node->parent()->y())
            );
        }
    }

   color = constants::Display::to_sdl_color(constants::Display::GREEN);
	if (initial_path_.size(0) > 1) {
		SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
		for (index_type i = 0; i < initial_path_.size(0) - 1; i++) {
            SDL_RenderDrawLine(renderer,
                static_cast<index_type>(initial_path_[i][0].item<real_t>()),
                static_cast<index_type>(initial_path_[i][1].item<real_t>()),
                static_cast<index_type>(initial_path_[i + 1][0].item<real_t>()),
                static_cast<index_type>(initial_path_[i + 1][1].item<real_t>())
            );
		}
	}

	color = constants::Display::to_sdl_color(constants::Display::RED);
	SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
	SDL_RenderDrawPoint(renderer, static_cast<index_type>(frenet_point_.a), static_cast<index_type>(frenet_point_.b));
	for (index_type i = 0; i < trajectory_.size(0) - 1; i++) {
		SDL_RenderDrawLine(renderer,
			static_cast<index_type>(trajectory_[i][0].item<real_t>()),
			static_cast<index_type>(trajectory_[i][1].item<real_t>()),
			static_cast<index_type>(trajectory_[i + 1][0].item<real_t>()),
			static_cast<index_type>(trajectory_[i + 1][1].item<real_t>())
		);
	}

    color = constants::Display::to_sdl_color(constants::Display::WHITE);
    if (fov_points_.size(0) > 1) {
        std::vector<SDL_Point> points(fov_points_.size(0));
        for (index_type i = 0; i < fov_points_.size(0); i++) {
            points[i] = {
                static_cast<index_type>(fov_points_[i][0].item<real_t>()),
                static_cast<index_type>(fov_points_[i][1].item<real_t>())
            };
        }

        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
        SDL_RenderDrawLines(renderer, points.data(), points.size());
        SDL_RenderDrawLine(renderer, points.back().x, points.back().y, points[0].x, points[0].y);
    }

    index_type x = static_cast<index_type>(position_[0].item<real_t>());
    index_type y = static_cast<index_type>(position_[1].item<real_t>());
    index_type r = static_cast<index_type>(radius_);

    SDL_SetRenderDrawColor(renderer, color_.r, color_.g, color_.b, color_.a);

    index_type direction_x = x + static_cast<index_type>(r * 1.5f * std::cos(yaw_));
    index_type direction_y = y + static_cast<index_type>(r * 1.5f * std::sin(yaw_));
    SDL_RenderDrawLine(renderer, x, y, direction_x, direction_y);

    // Bresenham's circle algorithm with filling
    index_type offsetx = 0;
    index_type offsety = r;
    index_type d = r - 1;

    while (offsety >= offsetx) {
        const SDL_Point points[] = {
            {x + offsetx, y + offsety}, {x + offsetx, y - offsety},
            {x - offsetx, y + offsety}, {x - offsetx, y - offsety},
            {x + offsety, y + offsetx}, {x + offsety, y - offsetx},
            {x - offsety, y + offsetx}, {x - offsety, y - offsetx}
        };
        SDL_RenderDrawPoints(renderer, points, 8);

        if (d >= 2*offsetx) {
            d -= 2*offsetx + 1;
            offsetx += 1;
        }
        else if (d < 2*(r-offsety)) {
            d += 2*offsety - 1;
            offsety -= 1;
        }
        else {
            d += 2*(offsety - offsetx - 1);
            offsety -= 1;
            offsetx += 1;
        }
    }
}

} // namespace object
