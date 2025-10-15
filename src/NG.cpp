#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<std::vector<int>> expand_cpp(
    const std::vector<std::vector<int>>& initial_grid,
    int generations) {
    
    std::vector<std::vector<int>> current_grid = initial_grid;

    return current_grid;
}

PYBIND11_MODULE(NG, m) {
    m.def("Expand_Cpp", &expand_cpp,
          "Simulate multiple generations of Conway's Game of Life and return all intermediate states",
          py::arg("initial_grid"), py::arg("generations"));
}