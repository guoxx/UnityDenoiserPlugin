#pragma once

// -----------------------------------------------------------------------------
// Transfer function: PU
// -----------------------------------------------------------------------------

// Fit of PU2 curve normalized at 100 cd/m^2
// [Aydin et al., 2008, "Extending Quality Metrics to Full Luminance Range Images"]
__device__ static const float PU_A  =  1.41283765e+03f;
__device__ static const float PU_B  =  1.64593172e+00f;
__device__ static const float PU_C  =  4.31384981e-01f;
__device__ static const float PU_D  = -2.94139609e-03f;
__device__ static const float PU_E  =  1.92653254e-01f;
__device__ static const float PU_F  =  6.26026094e-03f;
__device__ static const float PU_G  =  9.98620152e-01f;
__device__ static const float PU_Y0 =  1.57945760e-06f;
__device__ static const float PU_Y1 =  3.22087631e-02f;
__device__ static const float PU_X0 =  2.23151711e-03f;
__device__ static const float PU_X1 =  3.70974749e-01f;

//const float HDR_Y_MAX = 65504.;
__device__ static float PU_NORM_SCALE = 0.318967163f;//1.f / pu_forward(HDR_Y_MAX);

inline __device__ float PU_Forward(float y)
{
    if (y <= PU_Y0)
        return PU_A * y;
    else if (y <= PU_Y1)
        return PU_B * pow(y, PU_C) + PU_D;
    else
        return PU_E * log(y + PU_F) + PU_G;
}

inline __device__ float PU_Inverse(float x)
{
  if (x <= PU_X0)
    return x / PU_A;
  else if (x <= PU_X1)
    return pow((x - PU_D) / PU_B, 1.f / PU_C);
  else
    return exp((x - PU_G) / PU_E) - PU_F;
}