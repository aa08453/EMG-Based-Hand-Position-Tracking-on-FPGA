#ifndef AP_FIXED_H
#define AP_FIXED_H

#include <cmath>
#include <iostream>

// Minimal mock for ap_fixed that uses float internally.
// W: Width (Total bits), I: Integer bits — ignored in mock (just float).
template <int W, int I> struct ap_fixed {
  float val;

  ap_fixed() : val(0) {}
  ap_fixed(float v) : val(v) {}
  ap_fixed(double v) : val((float)v) {}
  ap_fixed(int v) : val((float)v) {}

  // Cross-type constructor: allow ap_fixed<A,B> -> ap_fixed<C,D>
  template <int W2, int I2>
  ap_fixed(const ap_fixed<W2, I2>& other) : val(other.val) {}

  // Implicit conversion to float
  operator float() const { return val; }

  // Assignment from float, double, int
  ap_fixed &operator=(float v) { val = v; return *this; }
  ap_fixed &operator=(double v) { val = (float)v; return *this; }
  ap_fixed &operator=(int v) { val = (float)v; return *this; }

  // Cross-type assignment
  template <int W2, int I2>
  ap_fixed &operator=(const ap_fixed<W2, I2>& other) { val = other.val; return *this; }

  // Arithmetic operators (same type)
  ap_fixed operator+(const ap_fixed &v) const { return ap_fixed(val + v.val); }
  ap_fixed operator-(const ap_fixed &v) const { return ap_fixed(val - v.val); }
  ap_fixed operator*(const ap_fixed &v) const { return ap_fixed(val * v.val); }
  ap_fixed operator/(const ap_fixed &v) const { return ap_fixed(val / v.val); }

  // Arithmetic with float
  float operator+(float v) const { return val + v; }
  float operator-(float v) const { return val - v; }
  float operator*(float v) const { return val * v; }
  float operator/(float v) const { return val / v; }

  // Cross-type arithmetic: ap_fixed<W,I> op ap_fixed<W2,I2>
  template <int W2, int I2>
  float operator+(const ap_fixed<W2, I2> &v) const { return val + v.val; }
  template <int W2, int I2>
  float operator-(const ap_fixed<W2, I2> &v) const { return val - v.val; }
  template <int W2, int I2>
  float operator*(const ap_fixed<W2, I2> &v) const { return val * v.val; }
  template <int W2, int I2>
  float operator/(const ap_fixed<W2, I2> &v) const { return val / v.val; }

  // Reverse arithmetic (float op ap_fixed)
  friend float operator+(float f, const ap_fixed &a) { return f + a.val; }
  friend float operator-(float f, const ap_fixed &a) { return f - a.val; }
  friend float operator*(float f, const ap_fixed &a) { return f * a.val; }
  friend float operator/(float f, const ap_fixed &a) { return f / a.val; }

  // Compound assignment
  ap_fixed &operator+=(float v) { val += v; return *this; }
  ap_fixed &operator-=(float v) { val -= v; return *this; }
  ap_fixed &operator+=(const ap_fixed &v) { val += v.val; return *this; }
  ap_fixed &operator-=(const ap_fixed &v) { val -= v.val; return *this; }

  // Cross-type compound assignment
  template <int W2, int I2>
  ap_fixed &operator+=(const ap_fixed<W2, I2> &v) { val += v.val; return *this; }
  template <int W2, int I2>
  ap_fixed &operator-=(const ap_fixed<W2, I2> &v) { val -= v.val; return *this; }

  // Comparison operators
  bool operator<(const ap_fixed &v) const { return val < v.val; }
  bool operator>(const ap_fixed &v) const { return val > v.val; }
  bool operator<=(const ap_fixed &v) const { return val <= v.val; }
  bool operator>=(const ap_fixed &v) const { return val >= v.val; }
  bool operator==(const ap_fixed &v) const { return val == v.val; }
  bool operator!=(const ap_fixed &v) const { return val != v.val; }

  // Cross-type comparisons
  template <int W2, int I2>
  bool operator<(const ap_fixed<W2, I2> &v) const { return val < v.val; }
  template <int W2, int I2>
  bool operator>(const ap_fixed<W2, I2> &v) const { return val > v.val; }

  // Unary minus
  ap_fixed operator-() const { return ap_fixed(-val); }

  // Stream output
  friend std::ostream& operator<<(std::ostream& os, const ap_fixed &a) {
    os << a.val;
    return os;
  }
};

#endif