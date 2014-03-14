#include <math.h>
#include <iostream>

// #ifdef NO_OMP
//    #define omp_get_thread_num() 0
// #else
//    #include <omp.h>
// #endif

#include "corr.h"

using namespace std;


Corr::Corr(int N_, float * ar1, float * ar2, float * ar3, short norm_) {
    
  N         =  N_;
  ar1_mean  =  mean_no_zero(ar1);
  ar2_mean  =  mean_no_zero(ar2);

  norm = norm_;
  
  ar1_stdev  = stdev_no_zero(ar1, ar1_mean);
  ar2_stdev  = stdev_no_zero(ar2, ar2_mean);

  if ( norm==0)
    norm_factor = ar1_mean * ar2_mean;
  else if ( norm==1)
    norm_factor = ar1_stdev * ar2_stdev;
  else
    norm_factor = 1;
  
  correlate ( ar1,ar2,ar3);
}

void Corr::correlate(float * ar1, float * ar2, float * arC) 
{
  // #pragma omp parallel for shared(arC)
  for ( int phi=0; phi < N; phi++ ) 
  {
    float counts(0); // keep track of the number of good pairs
    for ( int i=0; i < N; i++) 
    {
      int j = i + phi;
      
      if (j >= N) 
        j -= N;
      
      if (ar1[i] > 0 && ar2[j] > 0) 
      {
        arC[phi] += ( ar1[i] - ar1_mean) *( ar2[j]-ar2_mean) ;
        counts += 1;
      }
    }
    arC[phi] = arC[phi] / (norm_factor *  counts);
  }
}


float Corr::mean_no_zero(float * ar) 
{
  float ar_mean(0);
  float counts(0);
  int i(0);
  while(i < N)
  {
    if(ar[i] > 0)
    {
      ar_mean += ar[i];
      counts ++;
    }
    i ++;
  }
  if(counts > 0)
    return ar_mean / counts;
  else
    return 0;
}

float Corr::stdev_no_zero( float * ar, float ar_mean)
{
// the array ar is already mean subtracted

  float ar_stdev(0);
  float counts(0);
  int i(0);
  while(i < N)
  {
    if(ar[i] > 0)
    {
      ar_stdev += ( ar[i]- ar_mean )* ( ar[i] - ar_mean ) ;
      counts ++;
    }
    i ++;
  }
  if(counts > 0)
    return sqrt( ar_stdev / counts );
  else
    return 0;
}


Corr::~Corr() {
// destructor
}


