/** \file tb2integer.hpp
 *  \brief Unlimited precision integers with basic operations.
 *
 */

#ifndef TB2ENTIERS_HPP_
#define TB2ENTIERS_HPP_

#include <iostream>
#include <string>
#include <algorithm>
#include <cassert>
#include <sstream>

// Note: __int128_t is a compiler extension available in GCC and Clang.
// It provides 128-bit integer arithmetic but is not truly "big integer".

struct BigInteger {
  __int128_t value; ///< the 128-bit integer value

  // Default constructor: initialize to 0.
  BigInteger() : value(0) {}

  // Constructor from double.
  BigInteger(double d_) : value(static_cast<__int128_t>(d_)) {}

  // Copy constructor.
  BigInteger(const BigInteger& other) : value(other.value) {}

  // Destructor (nothing needed for built-in types).
  ~BigInteger() {}

  // Assignment operator.
  BigInteger& operator=(const BigInteger& other) {
    value = other.value;
    return *this;
  }

  // Addition assignment.
  BigInteger& operator+=(const BigInteger& other) {
    value += other.value;
    return *this;
  }

  // Subtraction assignment.
  BigInteger& operator-=(const BigInteger& other) {
    value -= other.value;
    return *this;
  }

  // Multiplication assignment.
  BigInteger& operator*=(const BigInteger& other) {
    value *= other.value;
    return *this;
  }

  // Division assignment.
  BigInteger& operator/=(const BigInteger& other) {
    assert(other.value != 0 && "Division by zero");
    value /= other.value;
    return *this;
  }

  // Unary negation operator.
  const BigInteger operator-() const {
    BigInteger result;
    result.value = -value;
    return result;
  }

  // Binary addition operator.
  friend const BigInteger operator+(const BigInteger& left, const BigInteger& right) {
    BigInteger result;
    result.value = left.value + right.value;
    return result;
  }

  // Binary subtraction operator.
  friend const BigInteger operator-(const BigInteger& left, const BigInteger& right) {
    BigInteger result;
    result.value = left.value - right.value;
    return result;
  }

  // Binary multiplication operator.
  friend const BigInteger operator*(const BigInteger& left, const BigInteger& right) {
    BigInteger result;
    result.value = left.value * right.value;
    return result;
  }

  // Binary division operator.
  friend const BigInteger operator/(const BigInteger& left, const BigInteger& right) {
    assert(right.value != 0 && "Division by zero");
    BigInteger result;
    result.value = left.value / right.value;
    return result;
  }

  // Equality operator.
  friend bool operator==(const BigInteger& left, const BigInteger& right) {
    return left.value == right.value;
  }

  // Inequality operator.
  friend bool operator!=(const BigInteger& left, const BigInteger& right) {
    return left.value != right.value;
  }

  // Less than or equal operator.
  friend bool operator<=(const BigInteger& left, const BigInteger& right) {
    return left.value <= right.value;
  }

  // Greater than or equal operator.
  friend bool operator>=(const BigInteger& left, const BigInteger& right) {
    return left.value >= right.value;
  }

  // Less than operator.
  friend bool operator<(const BigInteger& left, const BigInteger& right) {
    return left.value < right.value;
  }

  // Greater than operator.
  friend bool operator>(const BigInteger& left, const BigInteger& right) {
    return left.value > right.value;
  }

  // Helper: Convert __int128_t to a string.
  std::string toString() const {
    if (value == 0)
      return "0";
    __int128_t temp = value;
    bool isNegative = temp < 0;
    if (isNegative)
      temp = -temp;
    std::string str;
    while (temp > 0) {
      int digit = static_cast<int>(temp % 10);
      str.push_back('0' + digit);
      temp /= 10;
    }
    if (isNegative)
      str.push_back('-');
    std::reverse(str.begin(), str.end());
    return str;
  }

  // Print function.
  void print(std::ostream& os) const {
    os << toString();
  }

  // Stream insertion operator.
  friend std::ostream& operator<<(std::ostream& os, const BigInteger& bi) {
    bi.print(os);
    return os;
  }

  // Helper: Convert string to __int128_t.
  static __int128_t fromString(const std::string& s) {
    bool isNegative = false;
    size_t start = 0;
    if (!s.empty() && (s[0] == '-' || s[0] == '+')) {
      isNegative = (s[0] == '-');
      start = 1;
    }
    __int128_t result = 0;
    for (size_t i = start; i < s.size(); ++i) {
      if (s[i] >= '0' && s[i] <= '9')
        result = result * 10 + (s[i] - '0');
      else
        break; // Stop parsing at the first non-digit.
    }
    return isNegative ? -result : result;
  }

  // Stream extraction operator.
  friend std::istream& operator>>(std::istream& is, BigInteger& bi) {
    std::string input;
    is >> input;
    bi.value = fromString(input);
    return is;
  }
};

#endif
