/*
 *  solidangle.cpp
 *	--------------
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


#include <stdlib.h>
#include <cmath>
#include <iostream>


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
void fastSAC(int num_pixels, double * theta, double * correction_factor) {
			
	for (int i = 0; i < num_pixels; i++) {
		// JAS: note that this excludes the constant_factor:
		//      double constant_factor = pixel_size * pixel_size / (z_dist * z_dist)
		//      which can be viewed as the zeroth correction to the solid angle (obtained if theta = 0)
		//      and means the scattering will still be in its original scale after correction
		correction_factor[i] = 1.0 / cos(theta[i])*cos(theta[i])*cos(theta[i]);
	}
}



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
double pixelSolidAngle(double ccords[4][3]) {
        
	double determinant;
	double denominator;
	double solid_angle[2]; // solid angles of the two plane triangles that form the pixel
	double total_solid_angle;
    
    // distances of pixel corners, index starts from upper left corner and goes
    // around clock-wise
    double corner_distances[4];
    for (int j = 0; j < 4; j++) {
		corner_distances[j] = sqrt(ccords[j][0]*ccords[j][0] +
		                           ccords[j][1]*ccords[j][1] +
		                           ccords[j][2]*ccords[j][2]);
	}
    
    // first triangle made up of upper left, upper right, and lower right corner
    // numerator in expression for solid angle of a plane triangle --
    // magnitude of triple product of first 3 corners
    determinant = fabs( ccords[0][0]*(ccords[1][1]*ccords[2][2]  - 
                                                  ccords[1][2]*ccords[2][1]) - 
                        ccords[0][1]*(ccords[1][0]*ccords[2][2]  - 
                                                  ccords[1][2]*ccords[2][0]) + 
                        ccords[0][2]*(ccords[1][0]*ccords[2][1]  - 
                                                  ccords[1][1]*ccords[2][0]) );
				   
    denominator = corner_distances[0]*corner_distances[1]*corner_distances[2] + 
                  corner_distances[2]*(ccords[0][0]*ccords[1][0]  + 
                                       ccords[0][1]*ccords[1][1]  + 
                                       ccords[0][2]*ccords[1][2]) + 
                  corner_distances[1]*(ccords[0][0]*ccords[2][0]  + 
                                       ccords[0][1]*ccords[2][1]  + 
                                       ccords[0][2]*ccords[2][2]) + 
                  corner_distances[0]*(ccords[1][0]*ccords[2][0]  + 
                                       ccords[1][1]*ccords[2][1]  + 
                                       ccords[1][2]*ccords[2][2]);

    solid_angle[0] = atan2(determinant, denominator);

    // If det > 0 and denom < 0 arctan2 returns < 0, so add PI
    if (solid_angle[0] < 0) {
    	solid_angle[0] += M_PI;
    }

    // second triangle made up of lower right, lower left, and upper left corner
    // numerator in expression for solid angle of a plane triangle -- 
    // magnitude of triple product of last 3 corners
    determinant = fabs( ccords[0][0]*(ccords[3][1]*ccords[2][2]  - 
                                                  ccords[3][2]*ccords[2][1]) - 
                        ccords[0][1]*(ccords[3][0]*ccords[2][2]  - 
                                                  ccords[3][2]*ccords[2][0]) + 
                        ccords[0][2]*(ccords[3][0]*ccords[2][1]  - 
                                                  ccords[3][1]*ccords[2][0]) );
                                              
    denominator = corner_distances[2]*corner_distances[3]*corner_distances[0] + 
                  corner_distances[2]*(ccords[0][0]*ccords[3][0]  + 
                                       ccords[0][1]*ccords[3][1]  + 
                                       ccords[0][2]*ccords[3][2]) + 
                  corner_distances[3]*(ccords[0][0]*ccords[2][0]  + 
                                       ccords[0][1]*ccords[2][1]  + 
                                       ccords[0][2]*ccords[2][2]) + 
                  corner_distances[0]*(ccords[3][0]*ccords[2][0]  +
                                       ccords[3][1]*ccords[2][1]  + 
                                       ccords[3][2]*ccords[2][2]);

    solid_angle[1] = atan2(determinant, denominator);

    // If det > 0 and denom < 0 arctan2 returns < 0, so add PI
    if (solid_angle[1] < 0) {
    	solid_angle[1] += M_PI;
    }

    total_solid_angle = 2*(solid_angle[0] + solid_angle[1]);
    
    return total_solid_angle;
}



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
void rigorousGridSAC(int num_pixels_s,
                     int num_pixels_f,
                     // double pixel_size,
                     double * s,
                     double * f,
                     double * p,
                     double * correction_factor) {
 
    // for each pixel...
    // OMP possible here
    int num_pixels = num_pixels_s * num_pixels_f;
    for (int i = 0; i < num_pixels; i++) {
	
    	// compute vector coordinates of pixel corners
    	// first index starts from upper left corner and goes around clockwise
    	// second index determines X=0/Y=1/Z=2 coordinate
    	double corner_coordinates[4][3];
    	
    	// determine the pixel location, "v"
        double v[3];
        int ns, nf;
        ns = i / num_pixels_s;
        nf = i % num_pixels_f;
        
        // v = ns * s + nf * f + p
        for (int j = 0; j < 3; j++) {
            v[j] = ns * s[j] + nf * f[j] + p[j];
        }
        
        // find the corners by adding/subtracting appropriate diagonal vectors
        double d1[3];
        double d2[3];
        for (int j = 0; j < 3; j++) {
            d1[j] = s[j] / 2.0 + f[j] / 2.0;
            d2[j] = s[j] / 2.0 - f[j] / 2.0;
        }

	    // compute the corner positions
	    // JAS says: only relative order matters
	    for (int j = 0; j < 3; j++) {
            corner_coordinates[0][j] = v[j] - d2[j]; // upper left corner
            corner_coordinates[1][j] = v[j] + d1[j]; // upper right corner
            corner_coordinates[2][j] = v[j] + d2[j]; // lower right corner
            corner_coordinates[3][j] = v[j]	- d1[j]; // lower left corner
	    }
		
		/* double constant_factor = pixel_size * pixel_size / (z_dist * z_dist)
		 *
		 * the constant_factor is essentially a conversion to the units of 
		 * solid angle -- it can be viewed as the zeroth correction for the
		 * azimuthally symmetric case (obtained if theta = 0).
		 * if one wants to keep the original scale the scattering was recorded
		 * in (e.g. ADU/pixel), one can divide by this constant_factor and still
		 * obtain the correct q-dependence for the scattering
		 */		
		
		// JAS: note that the statement below is no longer in use!
		// we keep the absolute solid angle scale when pixelSolidAngle
		// is NOT divided by constant_factor!
		//
    	// remove constant term to only get theta/phi dependent part of 
    	// solid angle correction for 2D pattern
        correction_factor[i] = pixelSolidAngle(corner_coordinates);  // / constant_factor;
    }
}



/*
 * Compute the solid angle correction from an explicit xyz representation
 * of each pixel
 */
void rigorousExplicitSAC(double * pixel_xyz,
                         double * correction_factor) {
                         
    // not implemented yet...
    // TJL to JAS : we can probably just cp what you had in cheetah, but that
    // involves the approximation that detector rotation is negligable, right?
	
	// JAS: should be possible, note that the azimuthally symmetric correction
	// does assume the detector is perfectly aligned in the vertical/horizontal
	// plane perpendicular to the x-ray beam (i.e. the x-ray beam travels along the normal of the detector plane),
	// but the rigorous correction does not! It rigorously calculates the solid
	// angle of a (in this case of two) plane triangle(s) as seen from the origin,
	// so given the origin is set to the scattering source it does still work for
	// detector rotations!
    
}



