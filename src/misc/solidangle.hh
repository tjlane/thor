// solidangle.hh
// header for solidangle.cpp

void fastSAC(int num_pixels, double * theta, double * correction_factor);

void rigorousGridSAC(int num_pixels_s, int num_pixels_f, double * s,
    double * f, double * p, double * correction_factor);
                     
void rigorousExplicitSAC(double * pixel_xyz, double * correction_factor);