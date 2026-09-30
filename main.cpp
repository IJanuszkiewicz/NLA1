#include "lis.h"
// lis.h defines a function-like macro `conj(x)` (as `x` in the
// non-complex build), which collides with std::conj and Eigen::numext::conj.
// LIS does not use it in its headers, so drop it here.
#ifdef conj
#undef conj
#endif
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <optional>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Here are all (i think) IO functions needed
// ==== Misha ====
std::optional<Eigen::MatrixXd> read_eigen_from_png(std::string path) {
  return {}; // return null when error
}

void write_eigen_as_png(Eigen::VectorXd image, int width, int height,
                        std::string path) {}

void write_eigen_as_mpx(Eigen::SparseMatrix<double> matrix, std::string path) {}

void write_eigen_as_mpx(Eigen::VectorXd vector, std::string path) {}

std::optional<LIS_MATRIX> read_lis_matrix_from_mpx(std::string path) {
  return {};
}

std::optional<LIS_VECTOR> read_lis_vector_from_mpx(std::string path) {
  return {};
}

void write_lis_as_png(LIS_VECTOR v, int width, int height) {}

int main(int argc, char *argv[]) {
  // ==== Misha ==== (tasks 1-3)
  // Read image + print size
  // Add noise + save
  // Reshape + norm
  //
  // ==== Igor ==== (tasks 4-7)
  // Eigen stuff
  //
  // ==== Aleandro ==== (tasks 8-9)
  // Lis stuff
  //
  // ==== Igor ==== (rest)
  // More Eigen stuff

  // --- Eigen demo ---
  MatrixXd m = MatrixXd::Random(3, 3);
  m = (m + MatrixXd::Constant(3, 3, 1.0)) * 10;
  std::cout << "m =" << std::endl << m << std::endl;
  VectorXd v(3);
  v << 1, 0, 0;
  std::cout << "m * v =" << std::endl << m * v << std::endl;

  // --- Lis demo: solve A x = b ---
  LIS_MATRIX A;
  LIS_VECTOR b, x;
  LIS_SOLVER solver;
  LIS_INT n = 8, gn, is, ie, i;

  lis_initialize(&argc, &argv);

  lis_matrix_create(LIS_COMM_WORLD, &A);
  lis_matrix_set_size(A, 0, n);
  lis_matrix_get_size(A, &n, &gn);
  lis_matrix_get_range(A, &is, &ie);
  for (i = is; i < ie; i++) {
    if (i > 0)
      lis_matrix_set_value(LIS_INS_VALUE, i, i - 1, -1.0, A);
    if (i < gn - 1)
      lis_matrix_set_value(LIS_INS_VALUE, i, i + 1, -1.0, A);
    lis_matrix_set_value(LIS_INS_VALUE, i, i, 2.0, A);
  }
  lis_matrix_set_type(A, LIS_MATRIX_CSR);
  lis_matrix_assemble(A);

  lis_vector_duplicate(A, &b);
  lis_vector_duplicate(A, &x);
  lis_vector_set_all(1.0, b);

  lis_solver_create(&solver);
  lis_solve(A, b, x, solver);
  lis_vector_print(x);

  lis_solver_destroy(solver);
  lis_matrix_destroy(A);
  lis_vector_destroy(b);
  lis_vector_destroy(x);
  lis_finalize();

  return 0;
}
