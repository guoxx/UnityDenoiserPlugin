#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <memory>

namespace UnityDenoisePlugin
{

class GPUFence;
class GPUTexture;

enum OptixDenoiseReadback {
    None = 0,
    Albedo = 1,
    Normal = 2,
    Flow = 3,
    Color = 4,
    PreviousOutput = 5,
};

struct OptixDenoiseConfig {
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t tileWidth;
    uint32_t tileHeight;
    uint32_t guideAlbedo;
    uint32_t guideNormal;
    uint32_t temporalMode;
};

struct OptixDenoiseImageData {
    ID3D12Resource* albedo = nullptr;
    ID3D12Resource* normal = nullptr;
    ID3D12Resource* flow = nullptr;

    ID3D12Resource* color = nullptr;
    ID3D12Resource* output = nullptr;

    OptixDenoiseReadback readback = OptixDenoiseReadback::None;
    ID3D12Resource* readbackTexture = nullptr;
};

class OptixDenoiseContext
{
public:
    OptixDenoiseContext( const OptixDenoiseConfig& cfg );
    ~OptixDenoiseContext() noexcept( false );

    void Denoise( const OptixDenoiseImageData& data );

private:
    void DenoiseInternal( const OptixDenoiseImageData& data );

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

    uint32_t m_renderSyncCounter = 0;
    std::shared_ptr<GPUFence> m_cudaWaitFence = {};
    std::shared_ptr<GPUFence> m_d3dWaitFence = {};

    std::shared_ptr<GPUTexture> m_colorTexture = {};
    std::shared_ptr<GPUTexture> m_outputTexture = {};
    std::shared_ptr<GPUTexture> m_previousOutputTexture = {};

    std::shared_ptr<GPUTexture> m_guideAlbedoTexture = {};
    std::shared_ptr<GPUTexture> m_guideNormalTexture = {};
    std::shared_ptr<GPUTexture> m_guideFlowTexture = {};
};

}  // namespace UnityDenoisePlugin