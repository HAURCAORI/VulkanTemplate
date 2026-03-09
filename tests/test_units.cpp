#include <doctest/doctest.h>

#include "utils/Units.h"

#include <cmath>

using namespace vkt::units;

// -- Named constants ---------------------------------------------------------

TEST_CASE("Units -- named constants in metres") {
    CHECK(std::abs(MM  - 1.0e-3)              < 1e-15);
    CHECK(std::abs(CM  - 1.0e-2)              < 1e-14);
    CHECK(std::abs(M   - 1.0)                 < 1e-15);
    CHECK(std::abs(KM  - 1.0e3)               < 1e-9);
    CHECK(std::abs(Mm  - 1.0e6)               < 1e-6);
    CHECK(std::abs(AU  - 1.495978707000e11)   < 1.0); // IAU 2012, sub-metre tolerance
    CHECK(std::abs(LY  - 9.460730472580800e15)< 1.0e6);
    CHECK(std::abs(PC  - 3.085677581491367e16)< 1.0e6);
    CHECK(std::abs(KPC - 3.085677581491367e19)< 1.0e9);
    CHECK(std::abs(MPC - 3.085677581491367e22)< 1.0e12);
}

// -- Floating-point literals -------------------------------------------------

TEST_CASE("Units -- floating-point literals convert correctly") {
    CHECK(std::abs(1.0_mm  -  MM)  < 1e-18);
    CHECK(std::abs(1.0_cm  -  CM)  < 1e-18);
    CHECK(std::abs(1.0_m   -  M)   < 1e-18);
    CHECK(std::abs(1.0_km  -  KM)  < 1e-12);
    CHECK(std::abs(1.0_Mm  -  Mm)  < 1e-9);
    CHECK(std::abs(1.0_AU  -  AU)  < 1.0);
    CHECK(std::abs(1.0_ly  -  LY)  < 1.0e6);
    CHECK(std::abs(1.0_pc  -  PC)  < 1.0e6);
    CHECK(std::abs(1.0_kpc -  KPC) < 1.0e9);
    CHECK(std::abs(1.0_Mpc -  MPC) < 1.0e12);
}

// -- Integer literals --------------------------------------------------------

TEST_CASE("Units -- integer literals convert correctly") {
    CHECK(std::abs(1_mm  -  MM)  < 1e-18);
    CHECK(std::abs(1_km  -  KM)  < 1e-12);
    CHECK(std::abs(1_AU  -  AU)  < 1.0);
    CHECK(std::abs(1_ly  -  LY)  < 1.0e6);
}

// -- Scale relationships -----------------------------------------------------

TEST_CASE("Units -- 1000 mm == 1 m") {
    CHECK(std::abs(1000.0_mm - 1.0_m) < 1e-12);
}

TEST_CASE("Units -- 1000 m == 1 km") {
    CHECK(std::abs(1000.0_m - 1.0_km) < 1e-9);
}

TEST_CASE("Units -- 1000 km == 1 Mm") {
    CHECK(std::abs(1000.0_km - 1.0_Mm) < 1e-6);
}

TEST_CASE("Units -- AU is approximately 8.317 light-minutes") {
    // Speed of light = 299792458 m/s; 1 AU / c = ~499.0 seconds = ~8.317 minutes.
    constexpr double c       = 299792458.0; // m/s
    constexpr double lightMin = c * 60.0;   // metres per minute
    const double minutes = AU / lightMin;
    CHECK(std::abs(minutes - 8.317) < 0.01);
}
