// solidangle.hh
// header for solidangle.cpp

void fastSAC(int num_pixels, float * theta, float * correction_factor);

void rigorousGridSAC(int num_pixels_s, int num_pixels_f, float * s,
    float * f, float * p, float * correction_factor);
                     
void rigorousExplicitSAC(float * pixel_xyz, float * correction_factor);