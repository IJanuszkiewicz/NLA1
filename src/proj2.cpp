#include "functions_IO.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
  std::cout << "proj2" << std::endl;
  auto img = read_eigen_from_png("src/deer.jpg");

  return 0;
}
