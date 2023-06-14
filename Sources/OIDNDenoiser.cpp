#include "OIDNDenoiser.h"
#include "Exception.h"
#include "Utils.h"
#include "RHI.h"

namespace UnityDenoiserPlugin
{

OIDNDenoiser::OIDNDenoiser( const OIDNDenoiserConfig& cfg )
{
    CUDA_CHECK( cudaStreamCreateWithFlags( &m_stream, cudaStreamNonBlocking ) );

    m_cudaWaitFence = std::make_shared<Fence>();
    m_d3dWaitFence = std::make_shared<Fence>();

    int cudaDeviceIds[ 1 ] = { -1 };
    cudaStream_t cudaStreams[ 1 ] = { m_stream };
    m_device = oidnNewCUDADevice( cudaDeviceIds, cudaStreams, 1 );
    oidnCommitDevice( m_device );

    m_colorTexture = std::make_shared<OIDNTexture>( m_device, cfg.imageWidth, cfg.imageHeight, 3 );
    m_outputTexture = std::make_shared<OIDNTexture>( m_device, cfg.imageWidth, cfg.imageHeight, 3 );
    if ( cfg.guideAlbedo )
    {
        m_albedoTexture = std::make_shared<OIDNTexture>( m_device, cfg.imageWidth, cfg.imageHeight, 3 );
    }
    if ( cfg.guideNormal )
    {
        m_normalTexture = std::make_shared<OIDNTexture>( m_device, cfg.imageWidth, cfg.imageHeight, 3 );
    }

    // Create a filter for denoising a beauty (color) image using optional auxiliary images too
    // This can be an expensive operation, so try not to create a new filter for every image!
    m_filter = oidnNewFilter( m_device, "RT" );  // generic ray tracing filter
    oidnSetFilterImage( m_filter, "color", m_colorTexture->oidnBuffer,
                        OIDN_FORMAT_FLOAT3, cfg.imageWidth, cfg.imageHeight, 0, 0, 0 );  // beauty
    oidnSetFilterImage( m_filter, "output", m_outputTexture->oidnBuffer,
                        OIDN_FORMAT_FLOAT3, cfg.imageWidth, cfg.imageHeight, 0, 0, 0 );  // denoised beauty
    if ( cfg.guideAlbedo )
    {
        oidnSetFilterImage( m_filter, "albedo", m_albedoTexture->oidnBuffer,
                            OIDN_FORMAT_FLOAT3, cfg.imageWidth, cfg.imageHeight, 0, 0, 0 );  // auxiliary
    }
    if ( cfg.guideNormal )
    {
        oidnSetFilterImage( m_filter, "normal", m_normalTexture->oidnBuffer,
                            OIDN_FORMAT_FLOAT3, cfg.imageWidth, cfg.imageHeight, 0, 0, 0 );  // auxiliary
    }
    // beauty image is HDR
    oidnSetFilterBool( m_filter, "hdr", true );
    if ( cfg.guideAlbedo || cfg.guideNormal )
    {
        if ( cfg.cleanAux || cfg.prefilterAux )
        {
            // auxiliary features is noise-free
            oidnSetFilterBool( m_filter, "cleanAux", true );
        }
    }
    oidnCommitFilter( m_filter );

    if ( cfg.prefilterAux )
    {
        if ( cfg.guideAlbedo )
        {
            m_albedoFilter = oidnNewFilter( m_device, "RT" );
            oidnSetFilterImage( m_albedoFilter, "output", m_albedoTexture->oidnBuffer,
                                OIDN_FORMAT_FLOAT3, cfg.imageWidth, cfg.imageHeight, 0, 0, 0 );
            oidnSetFilterImage( m_albedoFilter, "albedo", m_albedoTexture->oidnBuffer,
                                OIDN_FORMAT_FLOAT3, cfg.imageWidth, cfg.imageHeight, 0, 0, 0 );
            oidnCommitFilter( m_albedoFilter );
        }

        if ( cfg.guideNormal )
        {
            m_normalFilter = oidnNewFilter( m_device, "RT" );
            oidnSetFilterImage( m_normalFilter, "output", m_normalTexture->oidnBuffer,
                                OIDN_FORMAT_FLOAT3, cfg.imageWidth, cfg.imageHeight, 0, 0, 0 );
            oidnSetFilterImage( m_normalFilter, "normal", m_normalTexture->oidnBuffer,
                                OIDN_FORMAT_FLOAT3, cfg.imageWidth, cfg.imageHeight, 0, 0, 0 );
            oidnCommitFilter( m_normalFilter );
        }
    }

    // Check for errors
    const char* errorMessage;
    if ( oidnGetDeviceError( m_device, &errorMessage ) != OIDN_ERROR_NONE )
        LogError( errorMessage );
}

OIDNDenoiser::~OIDNDenoiser() noexcept( false )
{
    // Make sure the stream is done before we destroy the denoiser, otherwise we'll get a crash.
    oidnSyncDevice( m_device );

    cudaDeviceSynchronize();
    CUDA_CHECK( cudaStreamDestroy( m_stream ) );

    if ( m_albedoFilter )
        oidnReleaseFilter( m_albedoFilter );
    if ( m_normalFilter )
        oidnReleaseFilter( m_normalFilter );
    oidnReleaseFilter( m_filter );
    oidnReleaseDevice( m_device );
}

void OIDNDenoiser::DenoiseInternal()
{
    if ( m_albedoFilter )
        oidnExecuteFilterAsync( m_albedoFilter );

    if ( m_normalFilter )
        oidnExecuteFilterAsync( m_normalFilter );

    oidnExecuteFilterAsync( m_filter );

    // Check for errors
    const char* errorMessage;
    if ( oidnGetDeviceError( m_device, &errorMessage ) != OIDN_ERROR_NONE )
        LogError( errorMessage );
}

void OIDNDenoiser::Denoise( const OIDNDenoiserImageData& data )
{
    if ( data.color == nullptr || data.output == nullptr ) {
        return;
    }

    // Update input textures
    std::vector<ID3D12Resource*> inputBuffers = { data.color };
    std::vector<ID3D12Resource*> internalInputBuffers = { m_colorTexture->d3dBuffer };
    if ( data.albedo && m_albedoTexture )
    {
        inputBuffers.push_back( data.albedo );
        internalInputBuffers.push_back( m_albedoTexture->d3dBuffer );
    }
    if ( data.normal && m_normalTexture )
    {
        inputBuffers.push_back( data.normal );
        internalInputBuffers.push_back( m_normalTexture->d3dBuffer );
    }
    RHI::CopyFromUnityBuffers( inputBuffers.data(), internalInputBuffers.data(), (int)inputBuffers.size() );

    // Sync between D3D and CUDA
    RHI::D3DSignalCudaWait( m_stream, m_cudaWaitFence.get() );

    // Denoise
    DenoiseInternal();

    // Sync between CUDA and D3D
    RHI::CudaSignalD3DWait( m_stream, m_d3dWaitFence.get() );

    // Copy output
    ID3D12Resource* outputBuffers[] = { data.output };
    ID3D12Resource* internalOutputBuffers[] = { m_outputTexture->d3dBuffer };
    RHI::CopyToUnityBuffers( internalOutputBuffers, outputBuffers, 1 );
}

}