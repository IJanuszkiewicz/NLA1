#include "functions_IO.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>

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

  auto mat = v1 * v1.transpose() + v2 * v2.transpose();
  return mat * 255;
}

// TASK 9: add random fluctuations of colour in [-scale, scale]
Eigen::MatrixXd with_noise(const Eigen::MatrixXd &mat, double scale) {
  Eigen::MatrixXd noisy =
      mat + (Eigen::MatrixXd::Random(mat.rows(), mat.cols()) * scale);
  noisy = noisy.cwiseMax(0.0).cwiseMin(255.0);
  return noisy;
}

// TASKS 11-12: split the SVD A ~ C D^T keeping the k largest singular triplets
template <typename Derived>
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
make_cd(const Eigen::SVDBase<Derived> &svd, int k) {
  const auto &U = svd.matrixU();        // n x m (thin) or n x n (full)
  const auto &s = svd.singularValues(); // m = min(n, p), sorted descending
  const auto &V = svd.matrixV();        // p x m (thin) or p x p (full)

  k = std::clamp(k, 0, static_cast<int>(s.size()));

  Eigen::MatrixXd C = U.leftCols(k);
  Eigen::MatrixXd Dt = s.head(k).asDiagonal() * V.leftCols(k).transpose();

  // Report the size of c and d
  std::cout << "For k = " << k << std::endl;
  std::cout << "size of C: " << C.rows() << "x" << C.cols() << std::endl;
  std::cout << "size of D: " << Dt.transpose().rows() << "x"
            << Dt.transpose().cols() << std::endl;

  return {C, Dt.transpose()};
}

// TASK 12: build the compressed image C D^T and save it
void save_compressed(const Eigen::MatrixXd &c, const Eigen::MatrixXd &d, int k,
                     const std::string &path) {

  Eigen::MatrixXd compressed = c * d.transpose();

  write_eigen_as_png(
      compressed.reshaped(compressed.rows() * compressed.cols(), 1),
      compressed.rows(), compressed.cols(), path);
}

int main(int argc, char *argv[]) {
  std::cout.precision(10);

  // TASK 8 Build the checkerboard and report its Euclidean norm
  std::cout << "\nTASK 8:" << std::endl;
  std::cout << "Building a 400x400 black and white checkerboard" << std::endl;
  auto checkerboard = make_checkerboard(400);
  std::cout << "Euclidean norm of the checkerboard: " << checkerboard.norm()
            << std::endl;

  // TASK 9 Add noise in [-60, 60] and save the noisy image
  std::cout << "\nTASK 9:" << std::endl;
  std::cout << "Adding random noise in [-60, 60] to each pixel" << std::endl;
  auto noisy_checkerboard = with_noise(checkerboard, 60);
  write_eigen_as_png(noisy_checkerboard.reshaped(noisy_checkerboard.rows() *
                                                     noisy_checkerboard.cols(),
                                                 1),
                     noisy_checkerboard.rows(), noisy_checkerboard.cols(),
                     "outputs/proj2/task9/noisy.png");

  // TASK 10 SVD of the noisy image, report the two largest singular values
  std::cout << "\nTASK 10:" << std::endl;
  std::cout << "Computing the SVD of the noisy checkerboard" << std::endl;
  Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::NoQRPreconditioner> svd(
      noisy_checkerboard, Eigen::ComputeThinU | Eigen::ComputeThinV);
  std::cout << "Two largest singular values of the noisy image:" << std::endl
            << svd.singularValues()(Eigen::seq(0, 1)) << std::endl;

  // TASK 11 Build C and D and report their size
  std::cout << "\nTASK 11:" << std::endl;
  std::cout << "Building the truncation matrices C and D (k = 6 and k = 12)"
            << std::endl;
  auto [c6, d6] = make_cd(svd, 6);
  auto [c12, d12] = make_cd(svd, 12);

  // TASK 12 Build the compressed images and save them
  std::cout << "\nTASK 12:" << std::endl;
  std::cout << "Computing the compressed images C D^T (k = 6 and k = 12)"
            << std::endl;
  save_compressed(c6, d6, 6, "outputs/proj2/task12/compressed6.png");
  save_compressed(c12, d12, 12, "outputs/proj2/task12/compressed12.png");

  return 0;
}
