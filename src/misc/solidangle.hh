/*
 *  solidangle.hh
 *	-------------
 * Solid angle corrections for pixel array detectors.
 *
 * Written as part of Cheetah: Jonas Sellberg <sellberg@slac.stanford.edu>
 * Modified for odin:          TJ Lane <tjlane@stanford.edu>
 *
 * Citations
 * ---------
 * [1] van Oosterom, A. & Strackee, J. The solid angle of a plane triangle. IEEE
 * Trans Biomed Eng 30, 125–126 (1983).
 * [2] Sellberg and Loh. See documentation.
 *
 */


/* 
 * Azimuthally symmetrical correction. Fast approximation to 
 * `rigorousSolidAngleCorrection`. See [2].
 * 
 * Parameters
 * ----------
 * num_pixels :        the number of pixels
 * theta :             scattering angle for each pixel (sometimes called 2-theta)
 * correction_factor : multiply this by intensities to correct
 *
 * Implementation notes
 * --------------------
 * Currently excludes the constant_factor:
 * double constant_factor = pixel_size * pixel_size / (z_dist * z_dist)
 * which can be viewed as the zeroth correction to the solid angle (obtained if theta = 0)
 * and means the scattering will still be in its original scale after correction
 */
void fastSAC(int num_pixels, double * theta, double * correction_factor);


/*
 * Compute the solid angle of a single pixel, given the corner coordinates.
 *
 * Parameters
 * ----------
 * ccords :            corner coordinates for a pixel,
 *					   where the first index specifies the corner (currently clock-wise from 0 = upper left corner)
 *					   and the second index specifies direction in Cartesian coordinates (0 = X, 1 = Y, 2 = Z)
 *
 * Implementation notes
 * --------------------
 * the total solid angle of a pixel is split up to calculating the solid angle for two plane triangles (see [1]),
 * which means that ccords should be specified so that ccords[1] and ccords[3] are DIAGONAL corners, 
 * and ccords[0] and ccords[2] are the two corners that merges the two plane triangles.
 */
double pixelSolidAngle(double ccords[4][3]);


/* 
 * Rigorous solid angle correction, for a single element of a BasisGrid. See
 * the Odin documentation for how this representation works. See also the
 * rigorousExplicitSAC() function below for computing the SAC on an
 * explicit represenatation of the detector. See [1].
 * 
 * Parameters
 * ----------
 * num_pixels_s : number of px on grid in the 's' direction
 * num_pixels_f : number of px on grid in the 'f' direction
 * s : slow scan vector for grid
 * f : fast scan vector for grid
 * p : data origin for grid
 * correction_factor (OUTPUT) : the corrections get put here
 */
void rigorousGridSAC(int num_pixels_s, int num_pixels_f, double * s,
    double * f, double * p, double * correction_factor);


/*
 * Compute the solid angle correction from an explicit xyz representation
 * of each pixel
 */
void rigorousExplicitSAC(double * pixel_xyz, double * correction_factor);
