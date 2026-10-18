#include "functions_IO.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char *argv[]) {
  std::cout << "proj2" << std::endl;

  // ==== Aleandro ==== (tasks 1-4)
  // TASK 1
  auto img = read_eigen_from_png("src/deer.jpg");
  if (!img.has_value()) {
    return 1;
  }
  MatrixXd eigen_image = img.value();
  MatrixXd prod_with_transposed = eigen_image.transpose() * eigen_image;
  std::cout << "Norm of A^T * A: " << prod_with_transposed.norm() << std::endl;


  // TASK 2
  
  return 0;
}
