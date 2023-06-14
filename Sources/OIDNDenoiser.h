#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <OpenImageDenoise/oidn.h>
#include <memory>
#include <string>
#include <vector>
#include "Utils.h"

namespace UnityDenoiserPlugin
{

class Fence;
class OIDNTexture;

struct OIDNDenoiserConfig {
    uint32_t imageWidth;
    uint32_t imageHeight;
    bool guideAlbedo;
    bool guideNormal;
    bool cleanAux;
    bool prefilterAux;
};

struct OIDNDenoiserImageData {
    ID3D12Resource* albedo = nullptr;
    ID3D12Resource* normal = nullptr;
    ID3D12Resource* color = nullptr;
    ID3D12Resource* output = nullptr;
};

class OIDNDenoiser
{
public:
    OIDNDenoiser( const OIDNDenoiserConfig& cfg );
    ~OIDNDenoiser() noexcept( false );

    void Denoise( const OIDNDenoiserImageData& data );

private:
    void DenoiseInternal();

    CUstream m_stream = nullptr;

    OIDNDevice m_device = nullptr;
    OIDNFilter m_filter = nullptr;
    OIDNFilter m_albedoFilter = nullptr;
    OIDNFilter m_normalFilter = nullptr;

    std::shared_ptr<Fence> m_cudaWaitFence = nullptr;
    std::shared_ptr<Fence> m_d3dWaitFence = nullptr;

    std::shared_ptr<OIDNTexture> m_colorTexture = nullptr;
    std::shared_ptr<OIDNTexture> m_albedoTexture = nullptr;
    std::shared_ptr<OIDNTexture> m_normalTexture = nullptr;
    std::shared_ptr<OIDNTexture> m_outputTexture = nullptr;
};

}  // namespace UnityDenoiserPlugin