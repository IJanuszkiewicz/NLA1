CXX      = mpicxx
CXXFLAGS = -DUSE_MPI -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX -I${mkEigenInc} -I${mkLisInc}
LDLIBS   = -L${mkLisLib} -llis

main: src/main.cpp src/functions_IO.cpp src/functions_IO.hpp
	$(CXX) $(CXXFLAGS) src/main.cpp src/functions_IO.cpp -o $@ $(LDLIBS)

run: main
	./main

clean:
	rm -rf main outputs

.PHONY: run clean
