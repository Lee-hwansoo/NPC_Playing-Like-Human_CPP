#ifndef EXAMPLE_HPP_
#define EXAMPLE_HPP_

#include <string>
#include <vector>

namespace example {

class Character {
public:
    Character(const std::string& name, int health);

    std::string getName() const;
    int getHealth() const;
    void takeDamage(int amount);
    void heal(int amount);

private:
    std::string m_name;
    int m_health;
};

// Utility functions
int calculateDamage(int baseDamage, int level);
std::vector<std::string> splitString(const std::string& str, char delimiter);

// Template function
template<typename T>
T clamp(T value, T min, T max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

}  // namespace example

#endif  // EXAMPLE_HPP_
