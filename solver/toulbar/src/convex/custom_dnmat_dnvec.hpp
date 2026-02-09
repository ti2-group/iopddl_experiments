#ifndef CUSTOM_DNMAT_DNVEC_HPP
#define CUSTOM_DNMAT_DNVEC_HPP

#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>


struct DnMat; // Forward declaration

//
// DnVec: a minimal dynamic column vector class.
//
struct DnVec {
  std::vector<Double> data;
  size_t size;

  DnVec() : size(0) {}
  DnVec(size_t n) : data(n, 0), size(n) {}

  // Create a zero vector of size n.
  static DnVec Zero(size_t n) {
    return DnVec(n);
  }

  // Create a vector of ones.
  static DnVec Ones(size_t n) {
    DnVec vec(n);
    std::fill(vec.data.begin(), vec.data.end(), 1);
    return vec;
  }

  // Create a constant vector.
  static DnVec Constant(size_t n, Double c) {
    DnVec vec(n);
    std::fill(vec.data.begin(), vec.data.end(), c);
    return vec;
  }

  // Element access.
  Double& operator()(size_t i) {
    assert(i < size);
    return data[i];
  }
  const Double& operator()(size_t i) const {
    assert(i < size);
    return data[i];
  }

  // Euclidean norm.
  Double norm() const {
    Double sum = 0;
    for (Double v : data)
      sum += v * v;
    return std::sqrt(sum);
  }

  // Squared norm.
  Double squaredNorm() const {
    Double sum = 0;
    for (Double v : data)
      sum += v * v;
    return sum;
  }

  // Dot product.
  Double dot(const DnVec &other) const {
    assert(size == other.size);
    Double sum = 0;
    for (size_t i = 0; i < size; i++)
      sum += data[i] * other.data[i];
    return sum;
  }

  // Arithmetic operators.
  DnVec operator+(const DnVec &other) const {
    assert(size == other.size);
    DnVec result(size);
    for (size_t i = 0; i < size; i++)
      result.data[i] = data[i] + other.data[i];
    return result;
  }
  DnVec operator-(const DnVec &other) const {
    assert(size == other.size);
    DnVec result(size);
    for (size_t i = 0; i < size; i++)
      result.data[i] = data[i] - other.data[i];
    return result;
  }
  DnVec operator*(Double scalar) const {
    DnVec result(size);
    for (size_t i = 0; i < size; i++)
      result.data[i] = data[i] * scalar;
    return result;
  }
  DnVec operator/(Double scalar) const {
    DnVec result(size);
    for (size_t i = 0; i < size; i++)
      result.data[i] = data[i] / scalar;
    return result;
  }

  DnVec& operator=(const DnVec &other) {
    if (this != &other) {
      size = other.size;
      data = other.data;
    }
    return *this;
  }

  // Return a new vector with the first n elements.
  DnVec head(size_t n) const {
    assert(n <= size);
    DnVec result(n);
    for (size_t i = 0; i < n; i++)
      result.data[i] = data[i];
    return result;
  }

  // Declaration of transpose: returns a 1 x size matrix.
  DnMat transpose() const;
};

//
// DnMat: a minimal dynamic dense matrix class.
//
struct DnMat {
  size_t rows, cols;
  std::vector<Double> data;

  DnMat() : rows(0), cols(0) {}
  DnMat(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0) {}

  // Create a zero matrix of size r x c.
  static DnMat Zero(size_t r, size_t c) {
    return DnMat(r, c);
  }

  // Element access.
  Double& operator()(size_t i, size_t j) {
    assert(i < rows && j < cols);
    return data[i * cols + j];
  }
  const Double& operator()(size_t i, size_t j) const {
    assert(i < rows && j < cols);
    return data[i * cols + j];
  }

  // Matrix addition.
  DnMat operator+(const DnMat &other) const {
    assert(rows == other.rows && cols == other.cols);
    DnMat result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
      result.data[i] = data[i] + other.data[i];
    return result;
  }
  // Matrix subtraction.
  DnMat operator-(const DnMat &other) const {
    assert(rows == other.rows && cols == other.cols);
    DnMat result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
      result.data[i] = data[i] - other.data[i];
    return result;
  }
  // Scalar multiplication.
  DnMat operator*(Double scalar) const {
    DnMat result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
      result.data[i] = data[i] * scalar;
    return result;
  }
  // Scalar division.
  DnMat operator/(Double scalar) const {
    DnMat result(rows, cols);
    for (size_t i = 0; i < data.size(); i++)
      result.data[i] = data[i] / scalar;
    return result;
  }
  // Matrix multiplication.
  DnMat operator*(const DnMat &other) const {
    assert(cols == other.rows);
    DnMat result(rows, other.cols);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < other.cols; j++) {
        Double sum = 0;
        for (size_t k = 0; k < cols; k++) {
          sum += (*this)(i, k) * other(k, j);
        }
        result(i, j) = sum;
      }
    }
    return result;
  }
  // Matrix-vector multiplication.
  DnVec operator*(const DnVec &vec) const {
    assert(cols == vec.size);
    DnVec result(rows);
    for (size_t i = 0; i < rows; i++) {
      Double sum = 0;
      for (size_t j = 0; j < cols; j++) {
        sum += (*this)(i, j) * vec.data[j];
      }
      result.data[i] = sum;
    }
    return result;
  }

  // Transpose of the matrix.
  DnMat transpose() const {
    DnMat result(cols, rows);
    for (size_t i = 0; i < rows; i++)
      for (size_t j = 0; j < cols; j++)
        result(j, i) = (*this)(i, j);
    return result;
  }

  // Trace (sum of diagonal entries).
  Double trace() const {
    assert(rows == cols);
    Double t = 0;
    for (size_t i = 0; i < rows; i++)
      t += (*this)(i, i);
    return t;
  }

  // Frobenius norm of the matrix.
  Double norm() const {
    Double sum = 0;
    for (Double v : data)
      sum += v * v;
    return std::sqrt(sum);
  }

  // Block proxy class to allow assignment to submatrices.
  class Block {
  public:
    DnMat &parent;
    size_t start_row, start_col, block_rows, block_cols;
    Block(DnMat &mat, size_t r, size_t c, size_t br, size_t bc)
        : parent(mat), start_row(r), start_col(c), block_rows(br), block_cols(bc) {}

    // Assignment from another DnMat.
    Block& operator=(const DnMat &sub) {
      assert(sub.rows == block_rows && sub.cols == block_cols);
      for (size_t i = 0; i < block_rows; i++)
        for (size_t j = 0; j < block_cols; j++)
          parent(start_row + i, start_col + j) = sub(i, j);
      return *this;
    }

    // Assignment from a DnVec (if the block is a column vector).
    Block& operator=(const DnVec &vec) {
      assert(block_cols == 1);
      assert(vec.size == block_rows);
      for (size_t i = 0; i < block_rows; i++)
        parent(start_row + i, start_col) = vec.data[i];
      return *this;
    }
  };

  // Return a block proxy for the submatrix starting at (r,c) with size (br x bc).
  Block block(size_t r, size_t c, size_t br, size_t bc) {
    assert(r + br <= rows && c + bc <= cols);
    return Block(*this, r, c, br, bc);
  }

  // Column proxy to simulate V.col(j) from Eigen.
  class Column {
  public:
    DnMat &parent;
    size_t colIndex;
    Column(DnMat &mat, size_t j) : parent(mat), colIndex(j) {}

    // Conversion to DnVec (extracts the column as a vector).
    operator DnVec() const {
      DnVec vec(parent.rows);
      for (size_t i = 0; i < parent.rows; i++)
        vec.data[i] = parent(i, colIndex);
      return vec;
    }

    // Assignment from a DnVec.
    Column& operator=(const DnVec &vec) {
      assert(vec.size == parent.rows);
      for (size_t i = 0; i < parent.rows; i++)
        parent(i, colIndex) = vec.data[i];
      return *this;
    }

    // Provide norm() for the column.
    Double norm() const {
      DnVec vec = *this;
      return vec.norm();
    }

    // Define operator/ for dividing the column by a scalar.
    DnVec operator/(Double scalar) const {
      DnVec vec = *this;
      for (size_t i = 0; i < parent.rows; i++) {
        vec.data[i] /= scalar;
      }
      return vec;
    }
  };

  // Non-const column access.
  Column col(size_t j) {
    assert(j < cols);
    return Column(*this, j);
  }

  // Const column extraction.
  DnVec col(size_t j) const {
    assert(j < cols);
    DnVec vec(rows);
    for (size_t i = 0; i < rows; i++)
      vec.data[i] = (*this)(i, j);
    return vec;
  }
};

// Non-member operator for scalar multiplication: scalar * DnMat.
inline DnMat operator*(Double scalar, const DnMat &mat) {
  return mat * scalar;
}

// Non-member operator for scalar multiplication: scalar * DnVec.
inline DnVec operator*(Double scalar, const DnVec &vec) {
  return vec * scalar;
}

// Now that DnMat is defined, implement DnVec::transpose().
inline DnMat DnVec::transpose() const {
  // Returns a 1 x size matrix (a row vector).
  DnMat result(1, size);
  for (size_t j = 0; j < size; j++)
    result(0, j) = data[j];
  return result;
}

#endif // CUSTOM_DNMAT_DNVEC_HPP
