#include <npc/example.hpp>

namespace example {

Character::Character(const std::string& name, int health)
    : m_name(name), m_health(health) {}

std::string Character::getName() const {
    return m_name;
}

int Character::getHealth() const {
    return m_health;
}

void Character::takeDamage(int amount) {
    m_health = std::max(0, m_health - amount);
}

void Character::heal(int amount) {
    m_health += amount;
}

int calculateDamage(int baseDamage, int level) {
    return baseDamage + (level * 2);  // 예시 구현
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : str) {
        if (c == delimiter) {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        } else {
            token += c;
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}

} // namespace example
