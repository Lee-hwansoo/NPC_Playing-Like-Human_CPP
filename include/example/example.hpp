/**
 * @file examples.hpp
 * @author HwanSoo Lee
 * @brief Example for Doxygen.
 * @details This file contains examples for Doxygen.
 * @version 1.0.0
 * @date 2023-12-28
 * @copyright Copyright (c) 2023
 */

#ifndef CPP_TEMPLATE_EXAMPLES_HPP_
#define CPP_TEMPLATE_EXAMPLES_HPP_

#include <cstdint>
#include <string>

namespace cpp_template::documentation {

/**
 * @brief Compare two numbers is equal.
 * @details This function compares two numbers is equal.
 * @param lhs Left hand side
 * @param rhs Right hand side
 * @return true if lhs is equal to rhs
 * @return false if lhs is not equal to rhs
 * @note Usage
 * @code{.cpp}
 * IsEqual(1.0, 1.0); // true
 * IsEqual(1.0, 2.0); // false
 * @endcode
 * @bug checking the case below
 * @code{.cpp}
 * IsEqual(0.5 + 0.5, 1.0) //false
 * @endcode
 * @todo Fix a bug case by Lee
 */
auto IsEqual(double lhs, double rhs) -> bool;

/**
 * @brief Add two numbers and return result.
 * @details This function add two numbers and return result.
 * @tparam DataType Type of data
 * @param lhs left hand side
 * @param rhs right hand side
 * @return DataType Result of addition
 */
template <typename DataType>
auto Add(DataType lhs, DataType rhs) -> DataType;

/**
 * @brief Divide two numbers and return result.
 * @details This function divides two numbers and return result.
 * @param lhs left hand side
 * @param rhs right hand side
 * @return double Result of division
 * @throw std::invalid_argument If rhs is zero, nan, inf or lhs is nan, inf
 */
auto Divide(double lhs, double rhs) -> double;

/**
 * @brief Point2 struct
 * @details This struct is used to store information about a point with
 * 2-Dimensional.
 * @tparam DataType Type of point coordinate
 * @author HwanSoo Lee
 * @version 1.0.0
 * @date 2023-12-28
 * @note This struct is used for 2D point.
 * @warning This struct is not used for 3D point.
 * @attention This struct is used for 2D point.
 * @deprecated until 1.1.0 This struct is deprecated. It will be changed to
 * Point2 class.
 */
template <typename DataType>
struct Point2 {
  DataType x{};  ///< x coordinate
  DataType y{};  ///< y coordinate
};

using Point2D = Point2<double>;  ///< 2D point with double type

/**
 * @brief Color enum class
 * @details This enum class is used to store information about a color.
 */
enum class Color {
  kRed = 0,    ///< Red color
  kGreen = 1,  ///< Green color
  kBlue = 2    ///< Blue color
};

/**
 * @brief Animal class
 * @details This class is used to store information about an animal.
 * @author HwanSoo Lee
 * @version 1.0.0
 * @date 2023-12-28
 */
class Animal {
 public:
  /**
   * @brief Basic constructor of Animal class is deleted.
   */
  Animal() = delete;

  /**
   * @brief Construct a new Animal object
   * @details This function constructs a new Animal object.
   * @param name Name of animal
   * @param age Age of animal
   */
  explicit Animal(std::string name, uint8_t age = 0);

  /**
   * @brief Copy constructor of Animal class is deleted.
   */
  Animal(const Animal &) = delete;

  /**
   * @brief Move constructor of Animal class is default.
   * @details This function moves a new Animal object.
   */
  Animal(Animal &&) = default;

  /**
   * @brief Copy assignment operator of Animal class is deleted.
   */
  auto operator=(const Animal &) -> Animal & = delete;

  /**
   * @brief Move assignment operator of Animal class is default.
   * @details This function moves a new Animal object.
   */
  auto operator=(Animal &&) -> Animal & = default;

  /**
   * @brief Destroy the Animal object
   * @details This function destroys the Animal object.
   */
  virtual ~Animal() = default;

  /**
   * @brief Virtual method Call animal sound
   * @details This function calls animal sound, but if is not implemented.
   * @return std::string Sound of animal
   */
  virtual auto Call() -> std::string;

 private:
  std::string name_{};  ///< Name of animal
  uint8_t age_{};       ///< Age of animal
};

/**
 * @brief Cat class
 * @details This class is used to store information about a cat.
 * @author HwanSoo Lee
 * @version 1.0.0
 * @date 2023-12-28
 */
class Cat : public Animal {
 public:
  /**
   * @brief Construct a new Cat object
   * @details This function constructs a new Cat object.
   * @param name Name of cat
   * @param age Age of cat
   */
  explicit Cat(std::string name, uint8_t age);

  /**
   * @brief Copy constructor of Cat class is deleted.
   */
  Cat(const Cat &) = delete;

  /**
   * @brief Move constructor of Cat class is default.
   * @details This function moves a new Cat object.
   */
  Cat(Cat &&) = default;

  /**
   * @brief Copy assignment operator of Cat class is deleted.
   */
  auto operator=(const Cat &) -> Cat & = delete;

  /**
   * @brief Move assignment operator of Cat class is default.
   * @details This function moves a new Cat object.
   */
  auto operator=(Cat &&) -> Cat & = default;

  /**
   * @brief Destroy the Cat object
   * @details This function destroys the Cat object.
   */
  ~Cat() override = default;

  auto Call() -> std::string override { return "Meow!"; }
};

}  // namespace cpp_template::documentation

#endif
