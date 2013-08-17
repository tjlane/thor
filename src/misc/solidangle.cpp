
/*
 * Solid angle corrections for pixel array detectors.
 *
 * Written:           Jonas Sellberg <sellberg@slac.stanford.edu>
 * Modified for odin: TJ Lane <tjlane@stanford.edu>
 *
 */


#include <stdlib.h>
#include <cmath>
#include <iostream>


void fastSAC(int num_pixels, float * theta, float * correction_factor) {
	/* 
	 * Azimuthally symmetrical correction. Fast approximation to 
	 * `rigorousSolidAngleCorrection`.
	 * 
	 * Parameters
	 * ----------
	 * num_pixels :        the number of pixels
	 * theta :             scattering angle theta for each pixel (IS THIS TWO-THETA?)
	 * correction_factor : multiply this by intensities to correct
	 */
			
	for (int i = 0; i < num_pixels; i++) {
		correction_factor[i] /= cos(theta[i])*cos(theta[i])*cos(theta[i]);
	}
}

		
void rigorousGridSAC(int num_pixels_s,
                     int num_pixels_f,
                     float * s,
                     float * f,
                     float * p,
                     float * correction_factor) {
    /* 
     * Rigorous solid angle correction, for a single element of a BasisGrid. See
     * the Odin documentation for how this representation works. See also the
     * rigorousExplicitSAC() function below for computing the SAC on an
     * explicit represenatation of the detector.
     * 
     * Parameters
     * ----------
     * num_pixels_s : number of px on grid in the 's' direction
     * num_pixels_f : number of px on grid in the 'f' direction
     * s : slow scan vector for grid
     * f : fast scan vector for grid
     * p : data origin for grid
     * correction_factor  (OUTPUT) : the corrections get put here
     */
 
    // for each pixel...
    // OMP possible here
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
	    // TJL to JAS : does the order here matter? Right now the comments are not *necessarily* correct...
	    for (int j = 0; j < 3; j++) {
        	corner_coordinates[0][j] = v[j] - d2[j] // upper left corner
        	corner_coordinates[1][j] = v[j] + d1[j] // upper right corner
        	corner_coordinates[2][j] = v[j] + d2[j] // lower right corner
        	corner_coordinates[3][j] = v[j]	- d1[j] // lower left corner
	    }
	
    	// remove constant term to only get theta/phi dependent part of 
    	// solid angle correction for 2D pattern
    	correction_factor[i] /= pixelSolidAngle(corner_coordinates) / solidAngle; // TJL to JAS : what is `solidAngle` here?, also might need to pass address...
    }
}

void rigorousExplicitSAC(float * pixel_xyz,
                         float * correction_factor) {
    /*
     * Compute the solid angle correction from an explicit xyz representation
     * of each pixel
     */
                         
    // not implemented yet...
    // TJL to JAS : we can probably just cp what you had in cheetah, but that
    // involves the approximation that detector rotation is negligable, right?
    
}


float pixelSolidAngle(float * corner_coordinates) {
    /*
     * Compute the solid angle of a single pixel, given the corner coordinates.
     *
     * Parameters
     * ----------
     * corner_coordinates : 
     */
        
	double determinant;
	double denominator;
	double solid_angle[2]; // solid angles of the two plane triangles that form the pixel
	double total_solid_angle;
    
    // distances of pixel corners, index starts from upper left corner and goes
    // around clock-wise
    double corner_distances[4];
    for (int j = 0; j < 4; j++) {
		corner_distances[j] = sqrt(corner_coordinates[j][0]*corner_coordinates[j][0] +
		                           corner_coordinates[j][1]*corner_coordinates[j][1] +
		                           corner_coordinates[j][2]*corner_coordinates[j][2]);
	}
    
    // first triangle made up of upper left, upper right, and lower right corner
    // numerator in expression for solid angle of a plane triangle --
    // magnitude of triple product of first 3 corners
    determinant = fabs( corner_coordinates[0][0]*(corner_coordinates[1][1]*corner_coordinates[2][2]  - 
                                                  corner_coordinates[1][2]*corner_coordinates[2][1]) - 
                        corner_coordinates[0][1]*(corner_coordinates[1][0]*corner_coordinates[2][2]  - 
                                                  corner_coordinates[1][2]*corner_coordinates[2][0]) + 
                        corner_coordinates[0][2]*(corner_coordinates[1][0]*corner_coordinates[2][1]  - 
                                                  corner_coordinates[1][1]*corner_coordinates[2][0]) );
				   
    denominator = corner_distances[0]*corner_distances[1]*corner_distances[2] + 
                  corner_distances[2]*(corner_coordinates[0][0]*corner_coordinates[1][0]  + 
                                       corner_coordinates[0][1]*corner_coordinates[1][1]  + 
                                       corner_coordinates[0][2]*corner_coordinates[1][2]) + 
                  corner_distances[1]*(corner_coordinates[0][0]*corner_coordinates[2][0]  + 
                                       corner_coordinates[0][1]*corner_coordinates[2][1]  + 
                                       corner_coordinates[0][2]*corner_coordinates[2][2]) + 
                  corner_distances[0]*(corner_coordinates[1][0]*corner_coordinates[2][0]  + 
                                       corner_coordinates[1][1]*corner_coordinates[2][1]  + 
                                       corner_coordinates[1][2]*corner_coordinates[2][2]);

    solid_angle[0] = atan2(determinant, denominator);

    // If det > 0 and denom < 0 arctan2 returns < 0, so add PI
    if (solid_angle[0] < 0) {
    	solid_angle[0] += M_PI;
    }

    // second triangle made up of lower right, lower left, and upper left corner
    // numerator in expression for solid angle of a plane triangle -- 
    // magnitude of triple product of last 3 corners
    determinant = fabs( corner_coordinates[0][0]*(corner_coordinates[3][1]*corner_coordinates[2][2]  - 
                                                  corner_coordinates[3][2]*corner_coordinates[2][1]) - 
                        corner_coordinates[0][1]*(corner_coordinates[3][0]*corner_coordinates[2][2]  - 
                                                  corner_coordinates[3][2]*corner_coordinates[2][0]) + 
                        corner_coordinates[0][2]*(corner_coordinates[3][0]*corner_coordinates[2][1]  - 
                                                  corner_coordinates[3][1]*corner_coordinates[2][0]) );
                                              
    denominator = corner_distances[2]*corner_distances[3]*corner_distances[0] + 
                  corner_distances[2]*(corner_coordinates[0][0]*corner_coordinates[3][0]  + 
                                       corner_coordinates[0][1]*corner_coordinates[3][1]  + 
                                       corner_coordinates[0][2]*corner_coordinates[3][2]) + 
                  corner_distances[3]*(corner_coordinates[0][0]*corner_coordinates[2][0]  + 
                                       corner_coordinates[0][1]*corner_coordinates[2][1]  + 
                                       corner_coordinates[0][2]*corner_coordinates[2][2]) + 
                  corner_distances[0]*(corner_coordinates[3][0]*corner_coordinates[2][0]  +
                                       corner_coordinates[3][1]*corner_coordinates[2][1]  + 
                                       corner_coordinates[3][2]*corner_coordinates[2][2]);

    solid_angle[1] = atan2(determinant, denominator);

    // If det > 0 and denom < 0 arctan2 returns < 0, so add PI
    if (solid_angle[1] < 0) {
    	solid_angle[1] += M_PI;
    }

    total_solid_angle = 2*(solid_angle[0] + solid_angle[1]);
    
    return total_solid_angle;
}
