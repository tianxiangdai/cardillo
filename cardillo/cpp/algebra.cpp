#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // For std::vector and other STL containers
#include <cmath>
#include <array>
#include <vector>
#include <cassert>

namespace py = pybind11;

// Constants
std::array<double, 3> e1 = {1.0, 0.0, 0.0};
std::array<double, 3> e2 = {0.0, 1.0, 0.0};
std::array<double, 3> e3 = {0.0, 0.0, 1.0};

// atan2 function
double atan2_custom(double y, double x) {
    if (x > 0) return std::atan(y / x);
    else if (x < 0) return y >= 0 ? std::atan(y / x) + M_PI : std::atan(y / x) - M_PI;
    else return y > 0 ? M_PI / 2 : y < 0 ? -M_PI / 2 : 0.0;
}

// ei function
std::array<double, 3> ei(int i) {
    std::array<double, 3> basis = e1;
    std::rotate(basis.begin(), basis.begin() + i % 3, basis.end());
    return basis;
}

// sign function
double sign(double x) {
    return std::copysign(1.0, x);
}

// norm function
double norm(const std::vector<double>& a) {
    double sum = 0.0;
    for (double v : a) sum += v * v;
    return std::sqrt(sum);
}

// LeviCivita3 function
int LeviCivita3(int i, int j, int k) {
    return (i - j) * (j - k) * (k - i) / 2;
}

// ax2skew function
py::array_t<double> ax2skew(const std::vector<double>& a) {
    assert(a.size() == 3);
    double data[3][3] = {
        {0, -a[2], a[1]},
        {a[2], 0, -a[0]},
        {-a[1], a[0], 0}
    };
    return py::array_t<double>({3, 3}, &data[0][0]);
}

// ax2skew_squared function
py::array_t<double> ax2skew_squared(const std::vector<double>& a) {
    assert(a.size() == 3);
    double a1 = a[0], a2 = a[1], a3 = a[2];
    double data[3][3] = {
        {-a2 * a2 - a3 * a3, a1 * a2, a1 * a3},
        {a2 * a1, -a1 * a1 - a3 * a3, a2 * a3},
        {a3 * a1, a3 * a2, -a1 * a1 - a2 * a2}
    };
    return py::array_t<double>({3, 3}, &data[0][0]);
}

// skew2ax function
py::array_t<double> skew2ax(py::array_t<double> A) {
    auto r = A.unchecked<2>();
    double data[3] = {0.5 * (r(2, 1) - r(1, 2)), 0.5 * (r(0, 2) - r(2, 0)), 0.5 * (r(1, 0) - r(0, 1))};
    return py::array_t<double>({{3}}, &data[0]);
}

// ax2skew_a function
py::array_t<double> ax2skew_a() {
    double data[3][3][3] = {{{0}}};
    data[1][2][0] = -1; data[2][1][0] = 1;
    data[0][2][1] = 1; data[2][0][1] = -1;
    data[0][1][2] = -1; data[1][0][2] = 1;
    return py::array_t<double>({3, 3, 3}, &data[0][0][0]);
}

// skew2ax_A function
py::array_t<double> skew2ax_A() {
    double data[3][3][3] = {{{0}}};
    data[0][2][1] = 0.5; data[0][1][2] = -0.5;
    data[1][0][2] = 0.5; data[1][2][0] = -0.5;
    data[2][1][0] = 0.5; data[2][0][1] = -0.5;
    return py::array_t<double>({3, 3, 3}, &data[0][0][0]);
}

// cross3 function
py::array_t<double> cross3(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == 3 && b.size() == 3);
    double data[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    return py::array_t<double>({{3}}, &data[0]);
}

// is_positive_definite function
// bool is_positive_definite(py::array_t<double> A) {
//     auto r = A.unchecked<2>();
//     int rows = r.shape(0);
//     assert(rows == r.shape(1));
//     for (int i = 1; i <= rows; ++i) {
//         py::array_t<double> sub_A(i, i);
//         if (py::detail::npy_format_descriptor<double>::det(sub_A) <= 0) return false;
//     }
//     return true;
// }

// Python module definition
PYBIND11_MODULE(algebra_cpp, m) {
    m.def("atan2", &atan2_custom);
    m.def("ei", &ei);
    m.def("sign", &sign);
    m.def("norm", &norm);
    m.def("LeviCivita3", &LeviCivita3);
    m.def("ax2skew", &ax2skew);
    m.def("ax2skew_squared", &ax2skew_squared);
    m.def("skew2ax", &skew2ax);
    m.def("ax2skew_a", &ax2skew_a);
    m.def("skew2ax_A", &skew2ax_A);
    m.def("cross3", &cross3);
    // m.def("is_positive_definite", &is_positive_definite);
}
