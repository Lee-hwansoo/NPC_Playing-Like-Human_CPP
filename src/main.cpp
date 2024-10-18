#include <npc/example.hpp>

#include <cstdint>
#include <iostream>
#include <cassert>

void testCharacter() {
  example::Character hero("Hero", 100);

  std::cout << "Testing Character class:\n";
  std::cout << "Name: " << hero.getName() << ", Health: " << hero.getHealth() << "\n";

  hero.takeDamage(20);
  std::cout << "After taking 20 damage, Health: " << hero.getHealth() << "\n";

  hero.heal(10);
  std::cout << "After healing 10, Health: " << hero.getHealth() << "\n";

  assert(hero.getName() == "Hero");
  assert(hero.getHealth() == 90);
}

void testCalculateDamage() {
    std::cout << "\nTesting calculateDamage function:\n";
    int damage = example::calculateDamage(10, 5);
    std::cout << "Calculated damage (base 10, level 5): " << damage << "\n";
    assert(damage > 10); // Assuming the function increases damage based on level
}

void testSplitString() {
  std::cout << "\nTesting splitString function:\n";
  std::string test = "Hello,World,OpenAI";
  std::vector<std::string> result = example::splitString(test, ',');

  std::cout << "Split result: ";
  for (const auto& str : result) {
      std::cout << str << " ";
  }
  std::cout << "\n";

  assert(result.size() == 3);
  assert(result[0] == "Hello");
  assert(result[1] == "World");
  assert(result[2] == "OpenAI");
}

void testClamp() {
  std::cout << "\nTesting clamp function:\n";

  int intResult = example::clamp(15, 0, 10);
  std::cout << "Clamp int (15, 0, 10): " << intResult << "\n";
  assert(intResult == 10);

  float floatResult = example::clamp(3.14f, 0.0f, 1.0f);
  std::cout << "Clamp float (3.14, 0.0, 1.0): " << floatResult << "\n";
  assert(floatResult == 1.0f);
}

int32_t main() {
  std::cout << "Starting tests for NPC project.\n\n";

  testCharacter();
  testCalculateDamage();
  testSplitString();
  testClamp();

  std::cout << "\nAll tests completed successfully!\n";
  return 0;
}