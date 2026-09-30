CXX      = mpicxx
CXXFLAGS = -DUSE_MPI -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX -I${mkEigenInc} -I${mkLisInc}
LDLIBS   = -L${mkLisLib} -llis

main: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $@ $(LDLIBS)

clean:
	rm -f main

.PHONY: clean
