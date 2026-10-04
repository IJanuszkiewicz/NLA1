#include "functions_IO.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "../libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb_image_write.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// creates missing parent directories of path (e.g. outputs/task2/)
static void make_parent_dirs(const std::string &path) {
  std::filesystem::path dir = std::filesystem::path(path).parent_path();
  if (!dir.empty()) {
    std::error_code ec; // on failure the following write reports the error
    std::filesystem::create_directories(dir, ec);
  }
}

// ==== Misha ====
std::optional<Eigen::MatrixXd> read_eigen_from_png(const std::string &path) {

  int width, height, channels;

  unsigned char *image_data =
      stbi_load(path.c_str(), &width, &height, &channels, 1);
  if (!image_data) {
    std::cerr << "Error: Could not load image " << path << std::endl;
    return {};
  }

  std::cout << "Image loaded: " << width << "x" << height << " with "
            << channels << " channels." << std::endl;
  MatrixXd img(height, width);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = (i * width + j);
      img(i, j) = static_cast<double>(image_data[index]);
    }
  }
  stbi_image_free(image_data);
  return img;
}
void write_eigen_as_png(const Eigen::MatrixXd &image, const std::string &path) {

  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      img(image.rows(), image.cols());
  img = image.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(
        std::clamp(val, 0.0, 255.0)); // clamp to avoid overflow
  });

  make_parent_dirs(path);
  if (stbi_write_png(path.c_str(), img.cols(), img.rows(), 1, img.data(),
                     img.cols()) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;
    return;
  }
  std::cout << "Image saved to " << path << std::endl;
}

void write_eigen_as_png(const Eigen::VectorXd &image, int width, int height,
                        const std::string &path) {
  MatrixXd matrix = image.reshaped<Eigen::RowMajor>(height, width);
  write_eigen_as_png(matrix, path);
}

void write_eigen_as_mtx(const Eigen::SparseMatrix<double> &matrix,
                        const std::string &path) {
  make_parent_dirs(path);
  saveMarket(matrix, path);
}

void write_eigen_as_mtx(const Eigen::VectorXd &vector,
                        const std::string &path) {
  // that was noted in the lab, the matrix mtx format from eigen is not
  // compatible with LIS, so write file manually. sparse matrix save works as it
  // is instead
  int size = vector.size();
  make_parent_dirs(path);
  FILE *out = fopen(path.c_str(), "w");
  if (!out) {
    std::cerr << "Error: could not open " << path << std::endl;
    return;
  }
  fprintf(out, "%%%%MatrixMarket vector coordinate real general\n");
  fprintf(out, "%d\n", size);
  for (int i = 0; i < size; i++) {
    fprintf(out, "%d %.17e\n", i + 1, vector(i));
  }
  fclose(out);
}

std::optional<LIS_MATRIX> read_lis_matrix_from_mtx(const std::string &path) {
  LIS_MATRIX matrix;
  lis_matrix_create(LIS_COMM_WORLD, &matrix);
  LIS_INT result = lis_input_matrix(
      matrix, const_cast<char *>(path.c_str())); // casting the type lis accepts
  if (result != LIS_SUCCESS) {
    return {};
  }
  return matrix;
}

std::optional<LIS_VECTOR> read_lis_vector_from_mtx(const std::string &path) {
  LIS_VECTOR vector;
  lis_vector_create(LIS_COMM_WORLD, &vector);
  LIS_INT result = lis_input_vector(
      vector, const_cast<char *>(path.c_str())); // casting the type LIS accepts
  if (result != LIS_SUCCESS) {
    return {};
  }
  return vector;
}

void write_lis_as_png(LIS_VECTOR v, int width, int height,
                      const std::string &path) {
  const int n = width * height;
  VectorXd vector(n);
  LIS_INT err = lis_vector_get_values(v, 0, n, vector.data());
  if (err != LIS_SUCCESS) {
    std::cerr << "Error: could not read LIS vector values" << std::endl;
    return;
  }

  write_eigen_as_png(vector, width, height, path);
}
