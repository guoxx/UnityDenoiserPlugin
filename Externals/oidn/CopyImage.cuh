#pragma once

#include <cuda_runtime.h>
#include <nvneural/CudaTypes.h>

enum ETransferFunction
{
    None,
    PU,
    PUInverse,
    EncodeNormal,
};

void CopyImage(CUstream stream,
               void *pDst, nvneural::TensorFormat dstFormat, nvneural::TensorDimension dstDimension,
               const void *pSrc, nvneural::TensorFormat srcFormat, nvneural::TensorDimension srcDimension,
               ETransferFunction transferFunction = ETransferFunction::None);
