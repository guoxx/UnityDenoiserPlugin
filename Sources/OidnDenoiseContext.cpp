#include "OidnDenoiseContext.h"

#include <optix.h>

#include "Exception.h"
#include "Utils.h"
#include "RHI.h"


namespace UnityDenoiserPlugin
{

void LogCallback( uint32_t level, const char* msg )
{
    if ( level == 2 )
        LogMessage( msg );
    else if ( level == 1 )
        LogWarning( msg );
    else
        LogError( msg );
}

OidnDenoiseContext::OidnDenoiseContext( const OidnDenoiseConfig& cfg )
{
    m_denoiser = std::make_shared<oidn::OidnDenoiser>( cfg.guideAlbedo, cfg.guideNormal, cfg.baseWeightsPath, cfg.pluginsFolder, LogCallback );

    m_cudaWaitFence = std::make_shared<GPUFence>( RHI::GetD3D12Device() );
    m_d3dWaitFence = std::make_shared<GPUFence>( RHI::GetD3D12Device() );

    m_colorTexture = std::make_shared<GPUTexture>( RHI::GetD3D12Device(), cfg.imageWidth, cfg.imageHeight, OPTIX_PIXEL_FORMAT_FLOAT4 );
    m_outputTexture = std::make_shared<GPUTexture>( RHI::GetD3D12Device(), cfg.imageWidth, cfg.imageHeight, OPTIX_PIXEL_FORMAT_FLOAT4 );

    if ( cfg.guideAlbedo )
    {
        m_guideAlbedoTexture = std::make_shared<GPUTexture>( RHI::GetD3D12Device(), cfg.imageWidth, cfg.imageHeight, OPTIX_PIXEL_FORMAT_FLOAT4 );
    }

    if ( cfg.guideNormal )
    {
        m_guideNormalTexture = std::make_shared<GPUTexture>( RHI::GetD3D12Device(), cfg.imageWidth, cfg.imageHeight, OPTIX_PIXEL_FORMAT_FLOAT4 );
    }
}

OidnDenoiseContext::~OidnDenoiseContext() noexcept( false )
{
    // Make sure the stream is done before we destroy the denoiser, otherwise we'll get a crash.
    cudaDeviceSynchronize();
}

oidn::Image2D ToOidnImage(OptixImage2D img)
{
    oidn::Image2D oi;
    oi.width = img.width;
    oi.height = img.height;
    oi.numChannels = 4;
    oi.dataType = oidn::DataType::Float;
    oi.dataLayout = oidn::DataLayout::NHWC;
    oi.data = (void*)img.data;
    return oi;
}

void OidnDenoiseContext::DenoiseInternal()
{
    oidn::Image2D colorImage = ToOidnImage(m_colorTexture->optixImage);
    oidn::Image2D outputImage = ToOidnImage(m_outputTexture->optixImage);

    oidn::Image2D albedoImage;
    if (m_guideAlbedoTexture)
        albedoImage = ToOidnImage(m_guideAlbedoTexture->optixImage);

    oidn::Image2D normalImage;
    if (m_guideNormalTexture)
        normalImage = ToOidnImage(m_guideNormalTexture->optixImage);

    m_denoiser->Denoise( colorImage,
                         m_guideAlbedoTexture ? &albedoImage : nullptr,
                         m_guideNormalTexture ? &normalImage : nullptr,
                         outputImage );
}

void OidnDenoiseContext::Denoise( const OidnDenoiseImageData& data )
{
    if ( data.color == nullptr || data.output == nullptr ) {
        return;
    }

    m_renderSyncCounter += 1;
    CUstream stream = m_denoiser->GetCudaStream();

    // Update input textures
    RHI::CopyContent( data.color, m_colorTexture->bufferHandle );
    if ( data.albedo ) {
        RHI::CopyContent( data.albedo, m_guideAlbedoTexture->bufferHandle );
    }
    if ( data.normal ) {
        RHI::CopyContent( data.normal, m_guideNormalTexture->bufferHandle );
    }

    // Sync between D3D and CUDA
    RHI::SignalD3D12Fence(m_cudaWaitFence->d3dFence, m_renderSyncCounter);
    WaitExternalSemaphore( m_cudaWaitFence->cudaSempaphore, m_renderSyncCounter, stream );

    // Denoise
    DenoiseInternal();

    // Wait CUDA to finish, for debugging
    // CUDA_SYNC_CHECK();

    // Sync between CUDA and D3D
    SignalExternalSemaphore( m_d3dWaitFence->cudaSempaphore, m_renderSyncCounter, stream );
    RHI::WaitD3D12Fence( m_d3dWaitFence->d3dFence, m_renderSyncCounter );

    // Copy output texture
    RHI::CopyContent( m_outputTexture->bufferHandle, data.output );

    // Copy readback texture
    if ( data.readback != Readback::None && data.readbackTexture != nullptr )
    {
        std::shared_ptr<GPUTexture> readbackTexture{};
        if ( data.readback == Readback::Albedo )
        {
            readbackTexture = m_guideAlbedoTexture;
        }
        else if ( data.readback == Readback::Normal )
        {
            readbackTexture = m_guideNormalTexture;
        }
        else if ( data.readback == Readback::Color )
        {
            readbackTexture = m_colorTexture;
        }

        if ( readbackTexture )
            RHI::CopyContent( readbackTexture->bufferHandle, data.readbackTexture );
    }
}

}