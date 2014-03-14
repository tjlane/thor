
class Corr
{
  int N;
  short norm;
  float ar1_mean, ar2_mean;
  float ar1_stdev, ar2_stdev;
  float norm_factor; 
  
  float mean_no_zero(float * ar);
  float stdev_no_zero(float * ar, float ar_mean );
  
  void correlate(float * ar1, float * ar2, float * ar3 );

public:
  Corr(int N_, float * ar1, float * ar2, float * ar3, short norm_);
  ~Corr();
};

