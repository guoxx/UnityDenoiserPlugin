#pragma once

#include <memory>
#include <cuda.h>
#include <string>

namespace oidn
{

    class OidnDenoiserImpl;

    enum DataType
    {
        Float,
        Half,
    };

    enum DataLayout
    {
        NCHW,
        NHWC,
    };

    struct Image2D
    {
        size_t width = 0;
        size_t height = 0;
        size_t numChannels = 0;
        DataType dataType = DataType::Float;
        DataLayout dataLayout = DataLayout::NHWC;
        void* data = nullptr;
    };

    typedef void (*LogCallback)(unsigned int level, const char *message);

    class OidnDenoiser
    {
    public:
        OidnDenoiser(bool guideAlbedo, bool guideNormal, const std::string& baseWeightsPath, const std::string& pluginsFolder, LogCallback logCallback = nullptr);

        CUstream GetCudaStream() const;

        void Denoise(const Image2D &colorImage, const Image2D* albedoImage, const Image2D* normalImage, const Image2D &outputImage) const;

    private:
        std::shared_ptr<OidnDenoiserImpl> impl;
    };

}