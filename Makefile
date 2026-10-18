CXX      = mpicxx
CXXFLAGS = -DUSE_MPI -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX -I${mkEigenInc} -I${mkLisInc}
LDLIBS   = -L${mkLisLib} -llis

main: src/main.cpp src/functions_IO.cpp src/functions_IO.hpp
	$(CXX) $(CXXFLAGS) src/main.cpp src/functions_IO.cpp -o $@ $(LDLIBS)

proj2: src/proj2.cpp src/functions_IO.cpp src/functions_IO.hpp
	$(CXX) $(CXXFLAGS) src/proj2.cpp src/functions_IO.cpp -o $@ $(LDLIBS)

run: main
	HWLOC_HIDE_ERRORS=2 ./main

run-proj2: proj2
	HWLOC_HIDE_ERRORS=2 ./proj2

clean:
	rm -rf main proj2 outputs

.PHONY: run run-proj2 clean
