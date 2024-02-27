#ifndef amici_spline_h
#define amici_spline_h

namespace amici {

#ifndef EXHALE_DOXYGEN_SHOULD_SKIP_THIS
int spline(
    int n, int end1, int end2, double slope1, double slope2, double x[],
    double y[], double b[], double c[], double d[]
);
#endif

double seval(
    int n, double u, double x[], double y[], double b[], double c[], double d[]
);

double sinteg(
    int n, double u, double x[], double y[], double b[], double c[], double d[]
);

} // namespace amici

#endif /* amici_spline_h */
