#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <memory>
#include "Utils.h"

namespace UnityDenoiserPlugin
{

class Fence;
class OptixTexture;

struct OptixDenoiserConfig {
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t tileWidth;
    uint32_t tileHeight;
    bool guideAlbedo;
    bool guideNormal;
    bool temporalMode;
};

struct OptixDenoiserImageData {
    ID3D12Resource* albedo = nullptr;
    ID3D12Resource* normal = nullptr;
    ID3D12Resource* flow = nullptr;
    ID3D12Resource* color = nullptr;
    ID3D12Resource* output = nullptr;
};

class OptixDenoiser_
{
public:
    OptixDenoiser_( const OptixDenoiserConfig& cfg );
    ~OptixDenoiser_() noexcept( false );

    void Denoise( const OptixDenoiserImageData& data );

private:
    void DenoiseInternal();

    bool m_firstFrame = true;
    bool m_temporalMode = false;
    unsigned int m_tileWidth = 0;
    unsigned int m_tileHeight = 0;
    unsigned int m_overlap = 0;

    CUdeviceptr m_intensity = 0;
    CUdeviceptr m_avgColor = 0;
    CUdeviceptr m_scratch = 0;
    uint32_t m_scratch_size = 0;
    CUdeviceptr m_state = 0;
    uint32_t m_state_size = 0;

    CUstream m_stream = nullptr;
    OptixDenoiser m_denoiser = nullptr;

    OptixDenoiserGuideLayer m_guideLayer;
    OptixDenoiserLayer m_denoiseLayer;

    std::shared_ptr<Fence> m_cudaWaitFence = nullptr;
    std::shared_ptr<Fence> m_d3dWaitFence = nullptr;

    std::shared_ptr<OptixTexture> m_colorTexture = nullptr;
    std::shared_ptr<OptixTexture> m_outputTexture = nullptr;
    std::shared_ptr<OptixTexture> m_previousOutputTexture = nullptr;

    std::shared_ptr<OptixTexture> m_guideAlbedoTexture = nullptr;
    std::shared_ptr<OptixTexture> m_guideNormalTexture = nullptr;
    std::shared_ptr<OptixTexture> m_guideFlowTexture = nullptr;

    static OptixDeviceContext s_optixDeviceContext;
};

}  // namespace UnityDenoisePlugin