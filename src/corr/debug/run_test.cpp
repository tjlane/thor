#include <fstream>
#include <iostream>
#include <time.h>
//#include "corr.h"
#include "corr.cpp"

using namespace std;


int main() {
    
    /* initialize random seed: */
    srand (time(NULL));
    
    int N = 10000;
    
    float * x = new float [N];
    float * y = new float [N];
    float * corr_out = new float [N];
    
    for ( int i=0; i<N; i++ ) {
        float fi = (float) i;
        x[i] =   0.1 * fi; // + ((double) rand() / (RAND_MAX));
        y[i] =   0.1 * fi; // + ((double) rand() / (RAND_MAX));
    }
    
    Corr corr_obj (N, x, y, corr_out);
    cout << "correlation: " << corr_out[0] << " "<< corr_out[1] << endl;
    
    delete [] x;
    delete [] y;
    delete [] corr_out;
    
    return 0;
    
}
