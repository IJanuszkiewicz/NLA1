#include "functions_IO.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

Eigen::SparseMatrix<double>
make_convolution_matrix(Eigen::MatrixXd h, int image_width, int image_height) {
  Eigen::SparseMatrix<double> a(image_width * image_height,
                                image_width * image_height);

  std::vector<Eigen::Triplet<double>> tripletList;
  tripletList.reserve(h.rows() * h.cols() * image_width);
  for (int row = 0; row < image_width * image_height; row++) {
    int out_row = row / image_width;
    int out_col = row % image_width;
    for (int i = 0; i < h.rows(); i++) {
      for (int j = 0; j < h.cols(); j++) {
        if (h(i, j) == 0.0) {
          continue;
        }
        int img_row = out_row - (h.rows() / 2) + i;
        int img_col = out_col - (h.cols() / 2) + j;
        if (img_row >= image_height || img_row < 0 || img_col >= image_width ||
            img_col < 0)
          continue;
        int a_col = img_row * image_width + img_col;
        tripletList.push_back(Eigen::Triplet<double>(row, a_col, h(i, j)));
      }
    }
  }
  a.setFromTriplets(tripletList.begin(), tripletList.end());
  return a;
}

bool is_symmetric(const Eigen::SparseMatrix<double> &a) {
  return a.isApprox(a.transpose(), 1e-15);
}

int main(int argc, char *argv[]) {
  std::cout.precision(10);

  // ==== Misha ==== (tasks 1-3)

  // TASK 1 Read image + print size
  std::cout << "\nTASK 1:" << std::endl;
  auto img = read_eigen_from_png("src/deer.jpg");

  if (!img.has_value()) {
    return 1;
  }

  MatrixXd eigen_image = img.value();

  std::cout << "Matrix size: " << eigen_image.rows() << "x"
            << eigen_image.cols() << " (m*n = " << eigen_image.size() << ")"
            << std::endl;

  // TASK 2 Add noise + save
  std::cout << "\nTASK 2:" << std::endl;
  MatrixXd noisy_eigen_image =
      eigen_image + (MatrixXd::Random(eigen_image.rows(), eigen_image.cols()) *
                     50); // creating noisy image
  noisy_eigen_image = noisy_eigen_image.cwiseMax(0.0).cwiseMin(
      255.0); // clamp noise so it doesnt overflow the range

  write_eigen_as_png(
      eigen_image,
      "outputs/task2/normal_output.png"); // saving also normal image, optional
  write_eigen_as_png(noisy_eigen_image, "outputs/task2/noisy_output.png");

  // TASK 3 Reshape + norm
  std::cout << "\nTASK 3:" << std::endl;
  VectorXd v = eigen_image.reshaped<Eigen::RowMajor>();
  VectorXd w = noisy_eigen_image.reshaped<Eigen::RowMajor>();

  std::cout << "Reshaped v: " << v.size()
            << " components (m*n = " << eigen_image.size() << ")" << std::endl;
  std::cout << "Reshaped w: " << w.size()
            << " components (m*n = " << eigen_image.size() << ")" << std::endl;

  double norm_v = v.norm();
  std::cout << "Euclidean norm of v: " << norm_v << std::endl;

  // ==== Igor ==== (tasks 4-7)
  // TASK 4
  std::cout << "\nTASK 4:" << std::endl;
  std::cout << "Building smoothing matrix A1" << std::endl;
  Eigen::Matrix3d hav1{{1, 1, 1}, {1, 4, 1}, {1, 1, 1}};
  hav1 *= 1.0 / 12.0;
  auto a1 =
      make_convolution_matrix(hav1, eigen_image.cols(), eigen_image.rows());
  std::cout << "A1 non-zeros: " << a1.nonZeros() << std::endl;
  std::cout << "A1 symmetric: " << (is_symmetric(a1) ? "yes" : "no")
            << std::endl;

  // TASK 5
  std::cout << "\nTASK 5:" << std::endl;
  std::cout << "Applying smoothing filter A1*w" << std::endl;
  auto smooth_w = a1 * w;
  write_eigen_as_png(smooth_w, eigen_image.cols(), eigen_image.rows(),
                     "outputs/task5/smooth_w.png");

  // TASK 6
  std::cout << "\nTASK 6:" << std::endl;
  std::cout << "Building sharpening matrix A2" << std::endl;
  Eigen::Matrix3d hsh1{{0, -3, 0}, {-1, 9, -3}, {0, -1, 0}};
  auto a2 =
      make_convolution_matrix(hsh1, eigen_image.cols(), eigen_image.rows());
  std::cout << "A2 non-zeros: " << a2.nonZeros() << std::endl;
  std::cout << "A2 symmetric: " << (is_symmetric(a2) ? "yes" : "no")
            << std::endl;

  // TASK 7
  std::cout << "\nTASK 7:" << std::endl;
  std::cout << "Applying sharpening filter A2*v" << std::endl;
  auto sharpened = a2 * v;

  write_eigen_as_png(sharpened, eigen_image.cols(), eigen_image.rows(),
                     "outputs/task7/sharpened.png");

  // ==== Aleandro ==== (tasks 8-9)
  // TASK 8
  std::cout << "\nTASK 8:" << std::endl;
  write_eigen_as_mtx(a2, "outputs/task8/A2.mtx");
  write_eigen_as_mtx(w, "outputs/task8/w.mtx");
  std::cout << "Solving A2 x = w with LIS: BiCGSTAB + ILU, tolerance 1e-12"
            << std::endl;
  lis_initialize(&argc, &argv);
  auto lis_A = read_lis_matrix_from_mtx("outputs/task8/A2.mtx");
  auto lis_b = read_lis_vector_from_mtx("outputs/task8/w.mtx");
  if (!lis_A || !lis_b) {
    return 1;
  }

  LIS_VECTOR lis_x;
  lis_vector_duplicate(lis_A.value(), &lis_x);
  lis_vector_set_all(0.0, lis_x); // initial guess

  LIS_SOLVER lis_solver;
  lis_solver_create(&lis_solver);
  char opts[] = "-i bicgstab -p ilu -tol 1e-12 -maxiter 5000";
  lis_solver_set_option(opts, lis_solver);

  lis_solve(lis_A.value(), lis_b.value(), lis_x, lis_solver);

  LIS_INT iters;
  double resid;
  lis_solver_get_iter(lis_solver, &iters);
  lis_solver_get_residualnorm(lis_solver, &resid);
  std::cout << "Iterations: " << iters << std::endl;
  std::cout << "Final residual: " << resid << " (tolerance 1e-12)" << std::endl;

  // TASK 9
  std::cout << "\nTASK 9:" << std::endl;
  std::cout << "Converting LIS solution x to png" << std::endl;
  write_lis_as_png(lis_x, eigen_image.cols(), eigen_image.rows(),
                   "outputs/task9/lis_solution.png");

  lis_solver_destroy(lis_solver);
  lis_matrix_destroy(lis_A.value());
  lis_vector_destroy(lis_b.value());
  lis_vector_destroy(lis_x);
  lis_finalize();

  // TASK 10
  std::cout << "\nTASK 10:" << std::endl;
  std::cout << "Building edge detection matrix A3" << std::endl;
  Eigen::Matrix3d hed2{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  auto a3 =
      make_convolution_matrix(hed2, eigen_image.cols(), eigen_image.rows());
  std::cout << "A3 non-zeros: " << a3.nonZeros() << std::endl;
  std::cout << "A3 symmetric: " << (is_symmetric(a3) ? "yes" : "no")
            << std::endl;

  // TASK 11
  std::cout << "\nTASK 11:" << std::endl;
  std::cout << "Applying edge detection filter A3*v" << std::endl;
  auto edge_detected = a3 * v;
  write_eigen_as_png(edge_detected, eigen_image.cols(), eigen_image.rows(),
                     "outputs/task11/edge_detected.png");

  // TASK 12
  std::cout << "\nTASK 12:" << std::endl;
  std::cout << "Solving (4I + A3) y = w with Eigen: BiCGSTAB, tolerance 1e-10"
            << std::endl;
  Eigen::SparseMatrix<double> identity(a3.rows(), a3.cols());
  identity.setIdentity();
  auto a4 = 4.0 * identity + a3;

  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
  solver.compute(a4);
  solver.setTolerance(1e-10);

  Eigen::VectorXd y(eigen_image.cols() * eigen_image.rows());

  y = solver.solve(w);
  std::cout << "Iterations: " << solver.iterations() << std::endl;
  std::cout << "Final residual: " << solver.error() << " (tolerance 1e-10)"
            << std::endl;
  write_eigen_as_png(y, eigen_image.cols(), eigen_image.rows(),
                     "outputs/task12/solved.png");

  return 0;
}
