#include "CopyImage.cuh"
#include <cuda_fp16.h>
#include <nvneural/CoreHelpers.h>
#include <nvneural/CudaHelpers.h>

#include "TransferFunction.cuh"

using namespace nvneural;

inline __device__ int getImageIndex(int x, int y, int c, int width, int height, int channels, bool isNHWC)
{
    int idx;

    if (isNHWC)
    {
        idx = (y * width + x) * channels + c;
    }
    else
    {
        idx = (c * height + y) * width + x;
    }

    return idx;
}

template <typename SrcType, typename DstType> DstType ConvertDataType(SrcType v);
template <> __device__ float ConvertDataType<float, float>(float v) { return v; }
template <> __device__ half ConvertDataType<half, half>(half v) { return v; }
template <> __device__ half ConvertDataType<float, half>(float v) { return __float2half(v); }
template <> __device__ float ConvertDataType<half, float>(half v) { return __half2float(v); }

template <typename T> __device__ float ToFloat(const T& value);
template <> __device__ float ToFloat<half>(const half& v) { return __half2float(v); }
template <> __device__ float ToFloat<float>(const float& v) { return v; }

template <typename T> __device__ T FromFloat(float v);
template <> __device__ half FromFloat<half>(float v) { return __float2half(v); }
template <> __device__ float FromFloat<float>(float v) { return v; }

template <typename T1, typename T2>
__global__ void CopyImageKernel(const void *pSrc_Typeless, nvneural::TensorDimension srcDimension, bool srcIsNHWC,
                                void *pDst_Typeless, nvneural::TensorDimension dstDimension, bool dstIsNHWC,
                                ETransferFunction transferFunc)
{
    const T1 *pSrc = (const T1 *)pSrc_Typeless;
    T2 *pDst = (T2 *)pDst_Typeless;
    const int dstWidth = dstDimension.w;
    const int dstHeight = dstDimension.h;
    const int dstChannels = dstDimension.c;
    const int srcWidth = srcDimension.w;
    const int srcHeight = srcDimension.h;
    const int srcChannels = srcDimension.c;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dstWidth && y < dstHeight)
    {
        for (int c = 0; c < dstChannels; c++)
        {
            int dst_idx = getImageIndex(x, y, c, dstWidth, dstHeight, dstChannels, dstIsNHWC);
            if (x < srcWidth && y < srcHeight && c < srcChannels)
            {
                int src_idx = getImageIndex(x, y, c, srcWidth, srcHeight, srcChannels, srcIsNHWC);

                float fVal = ToFloat<T1>(pSrc[src_idx]);
                if (transferFunc == ETransferFunction::PU)
                    fVal = PU_Forward(fVal) * PU_NORM_SCALE;
                else if (transferFunc == ETransferFunction::PUInverse)
                    fVal = PU_Inverse(fVal / PU_NORM_SCALE);
                else if (transferFunc == ETransferFunction::EncodeNormal)
                    fVal = fVal * 0.5 + 0.5;

                pDst[dst_idx] = FromFloat<T2>(fVal);
            }
            else
            {
                pDst[dst_idx] = static_cast<T2>(0); // Padding with black color (0) for the remaining destination region
            }
        }
    }
}

void CopyImage(CUstream stream,
               void *pDst, nvneural::TensorFormat dstFormat, nvneural::TensorDimension dstDimension,
               const void *pSrc, nvneural::TensorFormat srcFormat, nvneural::TensorDimension srcDimension,
               ETransferFunction transferFunction)
{

    dim3 dimBlock{32, 32, 1};
    dim3 blockGrid
    {
        static_cast<unsigned int>(DivideRoundingUp(dstDimension.w, dimBlock.x)),
        static_cast<unsigned int>(DivideRoundingUp(dstDimension.h, dimBlock.y)),
    };

    const bool dstIsNHWC = dstFormat.layout == nvneural::TensorDataLayout::Nhwc;
    const bool srcIsNHWC = srcFormat.layout == nvneural::TensorDataLayout::Nhwc;
    const bool dstIsFloat = dstFormat.elementType == nvneural::TensorDataType::Float;
    const bool srcIsFloat = srcFormat.elementType == nvneural::TensorDataType::Float;

    if (srcIsFloat)
    {
        if (dstIsFloat)
            CopyImageKernel<float, float><<<blockGrid, dimBlock, 0, stream>>>(pSrc, srcDimension, srcIsNHWC,
                                                                              pDst, dstDimension, dstIsNHWC,
                                                                              transferFunction);
        else
            CopyImageKernel<float, half><<<blockGrid, dimBlock, 0, stream>>>(pSrc, srcDimension, srcIsNHWC,
                                                                             pDst, dstDimension, dstIsNHWC,
                                                                             transferFunction);
    }
    else
    {
        if (dstIsFloat)
            CopyImageKernel<half, float><<<blockGrid, dimBlock, 0, stream>>>(pSrc, srcDimension, srcIsNHWC,
                                                                             pDst, dstDimension, dstIsNHWC,
                                                                             transferFunction);
        else
            CopyImageKernel<half, half><<<blockGrid, dimBlock, 0, stream>>>(pSrc, srcDimension, srcIsNHWC,
                                                                            pDst, dstDimension, dstIsNHWC,
                                                                            transferFunction);
    }
}
