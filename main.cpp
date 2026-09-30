#include "lis.h"
// lis.h defines a function-like macro `conj(x)` (as `x` in the
// non-complex build), which collides with std::conj and Eigen::numext::conj.
// LIS does not use it in its headers, so drop it here.
#ifdef conj
#undef conj
#endif
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <optional>
#include <string>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;

// Here are all (i think) IO functions needed
// ==== Misha ====
std::optional<Eigen::MatrixXd> read_eigen_from_png(const std::string& path) {

  int width, height, channels;
  
  unsigned char* image_data = stbi_load(path.c_str(), &width, &height, &channels, 1); 
  if (!image_data) {
    std::cerr << "Error: Could not load image " << path << std::endl;
    return {};
  }
 
  std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;
  MatrixXd img(height, width);

  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
      int index = (i*width + j);
      img(i,j) = static_cast<double>(image_data[index]);
    }  
  }
  stbi_image_free(image_data);
  return img;
}
void write_eigen_as_png(const Eigen::MatrixXd& image, const std::string& path) {

  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> img(image.rows(), image.cols());
  img = image.unaryExpr([](double val) -> unsigned char {
  return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));//clamp to avoid overflow
  });

  if (stbi_write_png(path.c_str(), img.cols(), img.rows(), 1, img.data(), img.cols() ) == 0){
    std::cerr << "Error: Could not save grayscale image" << std::endl;
    return;

  }
  std::cout << "Image saved to " << path << std::endl;
  }

void write_eigen_as_png(const Eigen::VectorXd& image, int width, int height, const std::string& path) {
  MatrixXd matrix = image.reshaped<Eigen::RowMajor>(height, width);
  write_eigen_as_png(matrix, path);
  }

void write_eigen_as_mtx(const Eigen::SparseMatrix<double>& matrix, const std::string& path) {
  saveMarket(matrix, path);
}

void write_eigen_as_mtx(const Eigen::VectorXd& vector, const std::string& path) {
  saveMarketVector(vector, path);
}

std::optional<LIS_MATRIX> read_lis_matrix_from_mtx(const std::string& path) { //TODO if needed
  return{};
}

std::optional<LIS_VECTOR> read_lis_vector_from_mtx(const std::string& path) { //TODO if needed
 return{};
}

void write_lis_as_png(LIS_VECTOR v, int width, int height, const std::string& path) {
  const int n = width * height;
  VectorXd vector(n);
  LIS_INT err = lis_vector_get_values(v, 0, n, vector.data());
  if (err != LIS_SUCCESS) {
    std::cerr << "Error: could not read LIS vector values" << std::endl;
    return;
  }

  write_eigen_as_png(vector, width, height, path);
}

int main(int argc, char *argv[]) {
  // ==== Misha ==== (tasks 1-3)
  // Read image + print size
  // Add noise + save
  // Reshape + norm
  auto img = read_eigen_from_png("./deer.jpg"); //reading image

  if (!img.has_value()) {
    std::cerr << "load failed" << std::endl;
    return 1;
  }

  MatrixXd eigen_image = img.value(); 

  std::cout << "Matrix size = " << eigen_image.rows() << "x" << eigen_image.cols() << " with " << eigen_image.size() << " elements" << std::endl;

  MatrixXd noisy_eigen_image = eigen_image + (MatrixXd::Random(eigen_image.rows(),eigen_image.cols())*50);  //creating noisy image
  noisy_eigen_image = noisy_eigen_image.cwiseMax(0.0).cwiseMin(255.0); // clamp noise so it doesnt overflow the range
  const std::string output_path = "./output.png"; 
  const std::string noise_output_path = "./noisy_output.png";

  write_eigen_as_png(eigen_image, output_path);
  write_eigen_as_png(noisy_eigen_image, noise_output_path); 

  VectorXd v = eigen_image.reshaped<Eigen::RowMajor>();// if you need default remove <Eigen::RowMajor>
  VectorXd w = noisy_eigen_image.reshaped<Eigen::RowMajor>(); // same

  std::cout << "Reshaped v rows " << v.rows() << " size " << v.size() << std::endl;
  std::cout << "Reshaped w rows " << w.rows() << " size " << w.size() << std::endl;

  double norm_v = v.norm();
  std::cout << "Norm of v " << norm_v << std::endl;

  write_eigen_as_png(w, eigen_image.cols(), eigen_image.rows(), "./write_eigen.png"); // check if writing is correct
  write_eigen_as_mtx(w, "write_eigen.mtx");
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
  /*MatrixXd m = MatrixXd::Random(3, 3);
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
  */
  return 0;
}
