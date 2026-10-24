#include "functions_IO.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <Eigen/SVD>

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;


// TASK 8: build a black and white checkerboard image
Eigen::MatrixXd make_checkerboard(int n) {
  int square_side(n / 10);

  Eigen::VectorXd v1(n);
  Eigen::VectorXd v2(n);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < square_side; j++) {
      v1(2 * i * square_side + j) = 1;
      v1((2 * i + 1) * square_side + j) = 0;
      v2(2 * i * square_side + j) = 0;
      v2((2 * i + 1) * square_side + j) = 1;
    }
  }

  auto mat = v1 * v2.transpose() + v2 * v1.transpose();
  return mat * 255;
}

// TASK 9: add random fluctuations of colour in [-scale, scale]
Eigen::MatrixXd with_noise(const Eigen::MatrixXd &mat, double scale) {
  Eigen::MatrixXd noisy =
      mat + (Eigen::MatrixXd::Random(mat.rows(), mat.cols()) * scale);
  noisy = noisy.cwiseMax(0.0).cwiseMin(255.0);
  return noisy;
}

// TASKS 6, 11-12: split the SVD A ~ C D^T keeping the k largest singular triplets
template <typename Derived>
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
make_cd(const Eigen::SVDBase<Derived> &svd, int k, bool task6 = false) {
  const auto &U = svd.matrixU();        // n x m (thin) or n x n (full)
  const auto &s = svd.singularValues(); // m = min(n, p), sorted descending
  const auto &V = svd.matrixV();        // p x m (thin) or p x p (full)

  k = std::clamp(k, 0, static_cast<int>(s.size()));

  Eigen::MatrixXd C = U.leftCols(k);
  Eigen::MatrixXd Dt = s.head(k).asDiagonal() * V.leftCols(k).transpose();

  std::cout << "For k = " << k << std::endl;
  if(task6){
    // Report the number of non zero values for task6
    std::cout << "Number of nonzero entries in C " << (C.array() != 0.0).count() << "\n"; 
    std::cout << "Number of nonzero entries in D " << (Dt.array() != 0.0).count() << "\n"; //nnz of Dt = nnz of D

  } else {
    // Report the size of c and d (for task 11)
    std::cout << "size of C: " << C.rows() << "x" << C.cols() << std::endl;
    std::cout << "size of D: " << Dt.transpose().rows() << "x"
            << Dt.transpose().cols() << std::endl;
  }
  

  return {C, Dt.transpose()};
}

// TASKS 7 and 12: build the compressed image C D^T and save it
void save_compressed(const Eigen::MatrixXd &c, const Eigen::MatrixXd &d, int k,
                     const std::string &path) {

  Eigen::MatrixXd compressed = c * d.transpose();
  write_eigen_as_png(
      compressed, path);
}






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
  std::cout << "Frobenius norm of A^T * A: " << ATA.norm() << std::endl;
  // NORM: 1.2105e+09

  // TASK 2
  std::cout << std::endl << "TASK 2" << std::endl;
  SelfAdjointEigenSolver<MatrixXd> eigensolver(ATA);
  if (eigensolver.info() != Eigen::Success) abort();
  std::cout << "The two largest singular value of A are:" << std::endl;
  std::cout << "1: " << std::sqrt(eigensolver.eigenvalues().tail(2)(1)) << std::endl;
  std::cout << "2: " << std::sqrt(eigensolver.eigenvalues().tail(2)(0)) << std::endl;
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

  // ==== Misha ==== (tasks 5-7)
  // TASK 5  SVD of A and report the Euclidean norm of sigma
  std::cout << "\nTASK 5" << std::endl;
  Eigen::BDCSVD<Eigen::MatrixXd> bdc_svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  VectorXd sigma = bdc_svd.singularValues();
  double sigma_norm = sigma.norm();
  std::cout << "Euclidian norm of diagonal matrix of singular values Sigma: " << sigma_norm << "\n";
  
  // TASK 6 Compute the matrices C and D described in (1) assuming k = 30 and k = 90. Report the
  //number of nonzero entries in the matrices C and D.
  std::cout << "\nTASK 6" << std::endl;
  auto [c30, d30] = make_cd(bdc_svd, 30, true);
  auto [c90, d90] = make_cd(bdc_svd, 90, true);

  // TASK 7
  std::cout << "\nTASK 7" << std::endl;
  std::cout << "Computing the compressed images C D^T (k = 30 and k = 90)"
            << std::endl;
  save_compressed(c30, d30, 30, "outputs/proj2/task7/deer_compressed30.png");
  save_compressed(c90, d90, 90, "outputs/proj2/task7/deer_compressed90.png");

  // ==== Igor ==== (tasks 8-12)
  // TASK 8 Build the checkerboard and report its Euclidean norm
  std::cout << "\nTASK 8" << std::endl;
  std::cout << "Building a 400x400 black and white checkerboard" << std::endl;
  auto checkerboard = make_checkerboard(400);
  std::cout << "Euclidean norm of the checkerboard: " << checkerboard.norm()
            << std::endl;

  // TASK 9 Add noise in [-60, 60] and save the noisy image
  std::cout << "\nTASK 9" << std::endl;
  std::cout << "Adding random noise in [-60, 60] to each pixel" << std::endl;
  auto noisy_checkerboard = with_noise(checkerboard, 60);
  
  write_eigen_as_png(noisy_checkerboard,
     "outputs/proj2/task9/noisy.png"); 

  // TASK 10 SVD of the noisy image, report the two largest singular values
  std::cout << "\nTASK 10" << std::endl;
  std::cout << "Computing the SVD of the noisy checkerboard" << std::endl;
  Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::NoQRPreconditioner> svd(
      noisy_checkerboard, Eigen::ComputeThinU | Eigen::ComputeThinV);
  std::cout << "Two largest singular values of the noisy image:" << std::endl
            << svd.singularValues()(Eigen::seq(0, 1)) << std::endl;

  // TASK 11 Build C and D and report their size
  std::cout << "\nTASK 11" << std::endl;
  std::cout << "Building the truncation matrices C and D (k = 6 and k = 12)"
            << std::endl;
  auto [c6, d6] = make_cd(svd, 6);
  auto [c12, d12] = make_cd(svd, 12);

  // TASK 12 Build the compressed images and save them
  std::cout << "\nTASK 12" << std::endl;
  std::cout << "Computing the compressed images C D^T (k = 6 and k = 12)"
            << std::endl;
  save_compressed(c6, d6, 6, "outputs/proj2/task12/compressed6.png");
  save_compressed(c12, d12, 12, "outputs/proj2/task12/compressed12.png");

  return 0;
}