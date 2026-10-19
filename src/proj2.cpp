#include "functions_IO.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;

int main(int argc, char *argv[]) {
  std::cout << "proj2" << std::endl;

  // ==== Aleandro ==== (tasks 1-4)
  // TASK 1
  auto img = read_eigen_from_png("src/deer.jpg");
  if (!img.has_value()) {
    return 1;
  }
  MatrixXd A = img.value();
  MatrixXd ATA = A.transpose() * A;
  std::cout << "Norm of A^T * A: " << ATA.norm() << std::endl << std::endl;
  // NORM: 1.2105e+09


  // TASK 2
  SelfAdjointEigenSolver<MatrixXd> eigensolver(ATA);
  if (eigensolver.info() != Eigen::Success) abort();
  std::cout << "The two largest eigenvalues of A^T * A are:\n" << eigensolver.eigenvalues().tail(2) << std::endl << std::endl;
  // EIGENVALUES: 1.12864e+08, 1.20029e+09


  // TASK 3
  SparseMatrix<double> ATA_sparse = ATA.sparseView();
  write_eigen_as_mtx(ATA_sparse, "./ATA.mtx");
  // Move into lis2.1.17_test
  // COMPILE: mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest1.c -o eigen1
  // RUN: mpirun -n 1 ./eigen1 ../ATA.mtx eigvec.txt hist.txt -e pi -etol 1e-10
  // RESULT: 1.200286e+09 ==> The result is coherent with the previous point
  // NOTE: It takes 10 iterations


// TASK 4
// Since the A^T * A matrix is SPD, the smallest eigenvalue is still positive. Therefore,
// a good shift might be floor(lambda_2 / 2) = 5.6e+07
//
// Move into lis2.1.17_test
// RUN: mpirun -n 1 ./eigen1 ../ATA.mtx eigvec.txt hist.txt -e pi -etol 1e-10 -shift 5.6e7
// RESULT: The number of iterations is now 9



  return 0;
}
