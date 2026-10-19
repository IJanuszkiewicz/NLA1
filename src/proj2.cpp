#include "functions_IO.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Eigenvalues>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;

int main(int argc, char *argv[]) {

  // ==== Aleandro ==== (tasks 1-4)
  // TASK 1
  std::cout << std::endl << "TASK 1" << std::endl;
  auto img = read_eigen_from_png("src/deer.jpg");
  if (!img.has_value()) {
    return 1;
  }
  MatrixXd A = img.value();
  std::cout << "A: " << A.rows() << " x " << A.cols()
            << ", min: " << A.minCoeff() << ", max: " << A.maxCoeff() << std::endl;
  MatrixXd ATA = A.transpose() * A;
  std::cout << "Frobenius norm of A^T * A: " << ATA.norm() << std::endl << std::endl;
  // NORM: 1.2105e+09


  // TASK 2
  std::cout << std::endl << "TASK 2" << std::endl;
  SelfAdjointEigenSolver<MatrixXd> eigensolver(ATA);
  if (eigensolver.info() != Eigen::Success) abort();
  std::cout << "The two largest singular value of A are:" << std::endl;
  std::cout << "1: " << std::sqrt(eigensolver.eigenvalues().tail(2)(1)) << std::endl;
  std::cout << "2: " << std::sqrt(eigensolver.eigenvalues().tail(2)(0)) << std::endl << std::endl;
  // EIGENVALUES: 1.20029e+09, 1.12864e+08
  // SINGULAR VALUES: 34645.1, 10623.8


  // TASK 3
  std::cout << std::endl << "TASK 3" << std::endl;
  SparseMatrix<double> ATA_sparse = ATA.sparseView();
  write_eigen_as_mtx(ATA_sparse, "lis2.1.17_test/ATA.mtx");
  // Move into lis2.1.17_test
  // COMPILE: mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest1.c -o eigen1
  // RUN: mpirun -n 1 ./eigen1 ATA.mtx eigvec.txt hist.txt -e pi -etol 1e-10
  // RESULT: 1.200286e+09 ==> The result is coherent with the previous point
  // NOTE: It takes 10 iterations


  // TASK 4
  // Since A^T * A is SPD, all eigenvalues except lambda_1 are in [0, lambda_2].
  // Shifting by mu = lambda_2 / 2 moves them into [-mu, mu]
  // So the method converges with ratio (lambda_2 / 2) / (lambda_1 - (lambda_2 / 2)) ~ 0.049
  // instead of lambda_2 / lambda_1 ~ 0.094.
  // mu = lambda_2 / 2 ~ 5.6e7
  //
  // Move into lis2.1.17_test
  // RUN: mpirun -n 1 ./eigen1 ATA.mtx eigvec.txt hist.txt -e pi -etol 1e-10 -shift 5.6e7
  // RESULT: The number of iterations is now 9



  return 0;
}
