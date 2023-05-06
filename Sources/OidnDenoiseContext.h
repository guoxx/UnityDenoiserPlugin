#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <OidnDenoiser.h>
#include "Utils.h"

namespace UnityDenoiserPlugin
{

class GPUFence;
class GPUTexture;

struct OidnDenoiseConfig {
    uint32_t imageWidth;
    uint32_t imageHeight;
    bool guideAlbedo;
    bool guideNormal;
    std::string baseWeightsPath;
    std::string pluginsFolder;
};

struct OidnDenoiseImageData {
    ID3D12Resource* albedo = nullptr;
    ID3D12Resource* normal = nullptr;

    ID3D12Resource* color = nullptr;
    ID3D12Resource* output = nullptr;

    Readback readback = Readback::None;
    ID3D12Resource* readbackTexture = nullptr;
};

class OidnDenoiseContext
{
public:
    OidnDenoiseContext( const OidnDenoiseConfig& cfg );
    ~OidnDenoiseContext() noexcept( false );

    void Denoise( const OidnDenoiseImageData& data );

private:
    void DenoiseInternal();

    std::shared_ptr<oidn::OidnDenoiser> m_denoiser;

    uint32_t m_renderSyncCounter = 0;
    std::shared_ptr<GPUFence> m_cudaWaitFence = {};
    std::shared_ptr<GPUFence> m_d3dWaitFence = {};

    std::shared_ptr<GPUTexture> m_guideAlbedoTexture = {};
    std::shared_ptr<GPUTexture> m_guideNormalTexture = {};
    std::shared_ptr<GPUTexture> m_colorTexture = {};
    std::shared_ptr<GPUTexture> m_outputTexture = {};
};

}  // namespace UnityDenoiserPlugin