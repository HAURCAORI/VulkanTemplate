#pragma once

// Distance unit literals and constants for large-scale (scientific) simulations.
// All values are in metres (the project's base world-coordinate unit).
//
// -- Quick start ---------------------------------------------------------------
//
//   #include "scene/Units.h"
//   using namespace vkt::units;
//
//   // Named constants
//   world.setWorldPosition(id, glm::dvec3(5.2 * AU, 0.0, 0.0)); // Jupiter ~5.2 AU
//
//   // User-defined literals (floating-point)
//   world.setWorldPosition(id, glm::dvec3(1.0_AU,  0.0, 0.0));
//   world.setWorldPosition(id, glm::dvec3(384400.0_km, 0.0, 0.0)); // Moon distance
//   camera.setPosition(glm::dvec3(4.244_ly, 0.0, 0.0));           // Alpha Centauri
//
//   // User-defined literals (integer)
//   world.setWorldPosition(id, glm::dvec3(1_AU, 0.0, 0.0));
//
// -- Scale reference -----------------------------------------------------------
//   1 mm  = 1e-3  m
//   1 cm  = 1e-2  m
//   1 km  = 1e3   m
//   1 Mm  = 1e6   m   (megametre, ~Earth radius x 1.57)
//   1 AU  = 1.496e11 m (Earth-Sun distance)
//   1 ly  = 9.461e15 m
//   1 pc  = 3.086e16 m (~3.26 ly)
//   1 kpc = 3.086e19 m

namespace vkt::units {

// -- Named constants (metres) --------------------------------------------------

inline constexpr double MM  = 1.0e-3;                  // millimetre
inline constexpr double CM  = 1.0e-2;                  // centimetre
inline constexpr double M   = 1.0;                     // metre (base unit)
inline constexpr double KM  = 1.0e3;                   // kilometre
inline constexpr double Mm  = 1.0e6;                   // megametre
inline constexpr double AU  = 1.495978707000e11;       // astronomical unit (IAU 2012)
inline constexpr double LY  = 9.460730472580800e15;    // light-year (Julian)
inline constexpr double PC  = 3.085677581491367e16;    // parsec
inline constexpr double KPC = 3.085677581491367e19;    // kiloparsec
inline constexpr double MPC = 3.085677581491367e22;    // megaparsec

// -- User-defined literals  --  floating-point (long double input) ---------------

constexpr double operator""_mm (long double v) { return static_cast<double>(v) * MM;  }
constexpr double operator""_cm (long double v) { return static_cast<double>(v) * CM;  }
constexpr double operator""_m  (long double v) { return static_cast<double>(v) * M;   }
constexpr double operator""_km (long double v) { return static_cast<double>(v) * KM;  }
constexpr double operator""_Mm (long double v) { return static_cast<double>(v) * Mm;  }
constexpr double operator""_AU (long double v) { return static_cast<double>(v) * AU;  }
constexpr double operator""_ly (long double v) { return static_cast<double>(v) * LY;  }
constexpr double operator""_pc (long double v) { return static_cast<double>(v) * PC;  }
constexpr double operator""_kpc(long double v) { return static_cast<double>(v) * KPC; }
constexpr double operator""_Mpc(long double v) { return static_cast<double>(v) * MPC; }

// -- User-defined literals  --  integer (unsigned long long input) ---------------

constexpr double operator""_mm (unsigned long long v) { return static_cast<double>(v) * MM;  }
constexpr double operator""_cm (unsigned long long v) { return static_cast<double>(v) * CM;  }
constexpr double operator""_m  (unsigned long long v) { return static_cast<double>(v) * M;   }
constexpr double operator""_km (unsigned long long v) { return static_cast<double>(v) * KM;  }
constexpr double operator""_Mm (unsigned long long v) { return static_cast<double>(v) * Mm;  }
constexpr double operator""_AU (unsigned long long v) { return static_cast<double>(v) * AU;  }
constexpr double operator""_ly (unsigned long long v) { return static_cast<double>(v) * LY;  }
constexpr double operator""_pc (unsigned long long v) { return static_cast<double>(v) * PC;  }
constexpr double operator""_kpc(unsigned long long v) { return static_cast<double>(v) * KPC; }
constexpr double operator""_Mpc(unsigned long long v) { return static_cast<double>(v) * MPC; }

} // namespace vkt::units
