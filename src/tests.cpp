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





  // TESTS WITH LIS
  lis_initialize(&argc, &argv);
  // DUMMY SPARSE MATRIX
  Eigen::SparseMatrix<double> test_A(5, 5);
  std::vector<Eigen::Triplet<double>> trip;
  for (int i = 0; i < 5; ++i) {
    trip.emplace_back(i, i, 2.0);
    if (i > 0)
      trip.emplace_back(i, i - 1, -1.0);
  }
  test_A.setFromTriplets(trip.begin(), trip.end());
  // WRITE AS MTX
  write_eigen_as_mtx(test_A, "test_A.mtx");

  auto lis_A = read_lis_matrix_from_mtx("test_A.mtx");
  if (!lis_A.has_value()) {
    std::cerr << "LIS matrix read failed" << std::endl;
  } else {
    LIS_INT n, gn;
    lis_matrix_get_size(lis_A.value(), &n, &gn);
    std::cout << "LIS matrix size: " << gn << std::endl;
    lis_matrix_destroy(lis_A.value());
  }

  // TEST write_eigen_as_mtx + read_lis_vector_from_mtx
  write_eigen_as_mtx(w, "w_test.mtx");

  auto lis_w = read_lis_vector_from_mtx("w_test.mtx");
  if (!lis_w.has_value()) {
    std::cerr << "LIS vector read failed" << std::endl;
  } else {
    write_lis_as_png(lis_w.value(), eigen_image.cols(), eigen_image.rows(),
                     "./write_read_write.png");
    lis_vector_destroy(lis_w.value());
  }

  lis_finalize();





  // BELOW ARE TESTS, REMOVE THEM WHEN YOU DONT NEED
  // TEST OF WRITING IMAGE AS .PNG AND .MTX
  write_eigen_as_png(w, eigen_image.cols(), eigen_image.rows(),
                     "./write_eigen.png"); // take vector v, provide dimensions
                                           // and write in provided path
  write_eigen_as_mtx(w,
                     "write_eigen.mtx"); // outputs the mtx file from matrix w
  */