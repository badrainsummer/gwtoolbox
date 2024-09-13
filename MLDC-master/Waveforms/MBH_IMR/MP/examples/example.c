#include "IMRPhenomD.h"

// Compile with
// gcc -o example example.c IMRPhenomD_internals.c IMRPhenomD.c -lgsl -lgslcblas

int main () {
    COMPLEX16FrequencySeries *htilde = NULL;
    const double phi0 = 0.0;
    const double fRef_in = 100.0;
    const double deltaF = 0.1;
    const double m1_SI = 10*MSUN_SI;
    const double m2_SI = 30*MSUN_SI;
    const double chi1 = 0.85;
    const double chi2 = 0.42;
    const double f_min = 10.0;
    const double f_max = 0.0;
    const double distance = 500.0e6 * PC_SI;

    int ret = IMRPhenomDGenerateFD(
        &htilde,    /**< [out] FD waveform */
        phi0,       /**< Orbital phase at fRef (rad) */
        fRef_in,    /**< reference frequency (Hz) */
        deltaF,     /**< Sampling frequency (Hz) */
        m1_SI,      /**< Mass of companion 1 (kg) */
        m2_SI,      /**< Mass of companion 2 (kg) */
        chi1,       /**< Aligned-spin parameter of companion 1 */
        chi2,       /**< Aligned-spin parameter of companion 2 */
        f_min,      /**< Starting GW frequency (Hz) */
        f_max,      /**< End frequency; 0 defaults to Mf = \ref f_CUT */
        distance    /**< Distance of source (m) */
    );

    // Note: to obtain \tilde h_+ and \tilde h_\times
    // multiply the returned frequency series `htilde` with
    // cos(inclination) for h+
    // -I * 0.5 * (1.0 + cos(inclination)^2) for hx

    for (int i=0; i<htilde->length; i++)
        printf("%g %g %g\n", i*htilde->deltaF, creal((htilde->data)[i]), cimag((htilde->data)[i]));

    DestroyCOMPLEX16FrequencySeries(htilde);
}
