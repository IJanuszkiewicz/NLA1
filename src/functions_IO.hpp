#pragma once

#include "lis.h"
// lis.h defines a function-like macro `conj(x)` (as `x` in the
// non-complex build), which collides with std::conj and Eigen::numext::conj.
// LIS does not use it in its headers, so drop it here.
#ifdef conj
#undef conj
#endif
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <optional>
#include <string>

std::optional<Eigen::MatrixXd> read_eigen_from_png(const std::string &path);

void write_eigen_as_png(const Eigen::MatrixXd &image, const std::string &path);
void write_eigen_as_png(const Eigen::VectorXd &image, int width, int height,
                        const std::string &path);

void write_eigen_as_mtx(const Eigen::SparseMatrix<double> &matrix,
                        const std::string &path);
void write_eigen_as_mtx(const Eigen::VectorXd &vector, const std::string &path);

std::optional<LIS_MATRIX> read_lis_matrix_from_mtx(const std::string &path);
std::optional<LIS_VECTOR> read_lis_vector_from_mtx(const std::string &path);

void write_lis_as_png(LIS_VECTOR v, int width, int height,
                      const std::string &path);
