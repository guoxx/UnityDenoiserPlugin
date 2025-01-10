#include "OptixDenoiser.h"

#include <optix.h>
#include <optix_denoiser_tiling.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

#include "Exception.h"
#include "Utils.h"
#include "RHI.h"


namespace UnityDenoiserPlugin
{

void LogCallback( uint32_t level, const char* tag, const char* message, void* /*cbdata*/ )
{
    std::ostringstream oss;
    oss << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";

    if ( level == 4 )
        LogMessage( oss.str().c_str() );
    else if ( level == 3 )
        LogWarning( oss.str().c_str() );
    else
        LogError( oss.str().c_str() );
}

OptixDeviceContext OptixDenoiser_::s_optixDeviceContext = nullptr;

OptixDenoiser_::OptixDenoiser_( const OptixDenoiserConfig& cfg )
{
    // Sanity check
    SUTIL_ASSERT_MSG( !cfg.guideNormal || cfg.guideAlbedo,
                      "Currently albedo is required if normal input is given" );
    SUTIL_ASSERT_MSG( ( cfg.tileWidth == 0 && cfg.tileHeight == 0 ) || ( cfg.tileWidth > 0 && cfg.tileHeight > 0 ),
                      "tile size must be > 0 for width and height" );

    m_temporalMode = cfg.temporalMode;
    m_tileWidth = cfg.tileWidth > 0 ? cfg.tileWidth : cfg.imageWidth;
    m_tileHeight = cfg.tileHeight > 0 ? cfg.tileHeight : cfg.imageHeight;

    m_cudaWaitFence = std::make_shared<Fence>();
    m_d3dWaitFence = std::make_shared<Fence>();

    // Initialize OptiX device context
    if (s_optixDeviceContext == nullptr)
    {
        // Initialize CUDA
        CUDA_CHECK( cudaFree( nullptr ) );

        OPTIX_CHECK( optixInit() );

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &LogCallback;
        options.logCallbackLevel = 4;
#if defined( DEBUG ) || defined( _DEBUG )
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif
        // zero means take the current context
        OPTIX_CHECK( optixDeviceContextCreate( nullptr, &options, &s_optixDeviceContext ) );
    }

    const bool kpMode = true;

    // Create denoiser
    {
        OptixDenoiserOptions options = {};
        options.guideAlbedo = cfg.guideAlbedo ? 1 : 0;
        options.guideNormal = cfg.guideNormal ? 1 : 0;
        options.denoiseAlpha = OptixDenoiserAlphaMode::OPTIX_DENOISER_ALPHA_MODE_COPY;

        OptixDenoiserModelKind modelKind;
        if ( kpMode ) {
            modelKind = cfg.temporalMode ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV : OPTIX_DENOISER_MODEL_KIND_AOV;
        }
        else {
            modelKind = cfg.temporalMode ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL : OPTIX_DENOISER_MODEL_KIND_HDR;
        }
        OPTIX_CHECK( optixDenoiserCreate( s_optixDeviceContext, modelKind, &options, &m_denoiser ) );
    }

    // Allocate device memory for denoiser
    {
        OptixDenoiserSizes denoiser_sizes;
        OPTIX_CHECK( optixDenoiserComputeMemoryResources( m_denoiser,
                                                          m_tileWidth,
                                                          m_tileHeight,
                                                          &denoiser_sizes ) );

        if ( cfg.tileWidth == 0 ) {
            m_scratch_size = static_cast<uint32_t>( denoiser_sizes.withoutOverlapScratchSizeInBytes );
            m_overlap = 0;
        } else {
            m_scratch_size = static_cast<uint32_t>( denoiser_sizes.withOverlapScratchSizeInBytes );
            m_overlap = denoiser_sizes.overlapWindowSizeInPixels;
        }
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_scratch ), m_scratch_size ) );

        m_state_size = static_cast<uint32_t>( denoiser_sizes.stateSizeInBytes );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_state ), denoiser_sizes.stateSizeInBytes ) );

        if ( kpMode ) {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_avgColor ), sizeof( float ) * 3 ) );
        } else {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_intensity ), sizeof( float ) ) );
        }

        // Create denoise layer resources
        {
            m_colorTexture = std::make_shared<OptixTexture>( cfg.imageWidth, cfg.imageHeight, 3 );
            m_outputTexture = std::make_shared<OptixTexture>( cfg.imageWidth, cfg.imageHeight, 3 );

            m_denoiseLayer = {};
            m_denoiseLayer.type = OPTIX_DENOISER_AOV_TYPE_NONE;
            m_denoiseLayer.input = m_colorTexture->optixImage;
            m_denoiseLayer.output = m_outputTexture->optixImage;

            if ( cfg.temporalMode )
            {
                m_previousOutputTexture = std::make_shared<OptixTexture>( cfg.imageWidth, cfg.imageHeight, 3 );
                m_denoiseLayer.previousOutput = m_previousOutputTexture->optixImage;
            }
        }

        // Create guide layer resources
        {
            m_guideLayer = {};

            if ( cfg.guideAlbedo )
            {
                m_guideAlbedoTexture = std::make_shared<OptixTexture>( cfg.imageWidth, cfg.imageHeight, 3 );
                m_guideLayer.albedo = m_guideAlbedoTexture->optixImage;
            }

            if ( cfg.guideNormal ) {
                m_guideNormalTexture = std::make_shared<OptixTexture>( cfg.imageWidth, cfg.imageHeight, 3 );
                m_guideLayer.normal = m_guideNormalTexture->optixImage;
            }

            if ( cfg.temporalMode )
            {
                m_guideFlowTexture = std::make_shared<OptixTexture>( cfg.imageWidth, cfg.imageHeight, 2 );
                m_guideLayer.flow = m_guideFlowTexture->optixImage;

                // Internal guide layer memory set to zero for first frame.
                void* internalMemIn = 0;
                void* internalMemOut = 0;
                size_t internalSize = cfg.imageWidth * cfg.imageHeight * denoiser_sizes.internalGuideLayerPixelSizeInBytes;
                CUDA_CHECK( cudaMalloc( &internalMemIn, internalSize ) );
                CUDA_CHECK( cudaMalloc( &internalMemOut, internalSize ) );
                CUDA_CHECK( cudaMemset( internalMemIn, 0, internalSize ) );
                CUDA_CHECK( cudaMemset( internalMemOut, 0, internalSize ) );

                m_guideLayer.previousOutputInternalGuideLayer.data = (CUdeviceptr)internalMemIn;
                m_guideLayer.previousOutputInternalGuideLayer.width = cfg.imageWidth;
                m_guideLayer.previousOutputInternalGuideLayer.height = cfg.imageHeight;
                m_guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes = unsigned( denoiser_sizes.internalGuideLayerPixelSizeInBytes );
                m_guideLayer.previousOutputInternalGuideLayer.rowStrideInBytes = m_guideLayer.previousOutputInternalGuideLayer.width * m_guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes;
                m_guideLayer.previousOutputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;

                m_guideLayer.outputInternalGuideLayer = m_guideLayer.previousOutputInternalGuideLayer;
                m_guideLayer.outputInternalGuideLayer.data = (CUdeviceptr)internalMemOut;
            }
        }
    }

    // Setup denoiser
    {
        OPTIX_CHECK( optixDenoiserSetup(
            m_denoiser,
            nullptr,  // CUDA stream
            m_tileWidth + 2 * m_overlap,
            m_tileHeight + 2 * m_overlap,
            m_state,
            m_state_size,
            m_scratch,
            m_scratch_size ) );
    }

    CUDA_CHECK( cudaStreamCreateWithFlags( &m_stream, cudaStreamNonBlocking ) );
}

OptixDenoiser_::~OptixDenoiser_() noexcept( false )
{
    // Make sure the stream is done before we destroy the denoiser, otherwise we'll get a crash.
    CUDA_CHECK( cudaStreamSynchronize( m_stream ) );
    CUDA_CHECK( cudaStreamDestroy( m_stream ) );

    OPTIX_CHECK( optixDenoiserDestroy( m_denoiser ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_intensity ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_avgColor ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_scratch ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_state ) ) );
}

void OptixDenoiser_::DenoiseInternal()
{
    OptixDenoiserParams denoiserParams = {};
    denoiserParams.hdrIntensity = m_intensity;
    denoiserParams.hdrAverageColor = m_avgColor;
    denoiserParams.blendFactor = 0.0f;
    denoiserParams.temporalModeUsePreviousLayers = m_firstFrame ? 0 : 1;

    if ( m_intensity )
    {
        OPTIX_CHECK( optixDenoiserComputeIntensity(
            m_denoiser,
            m_stream,
            &m_denoiseLayer.input,
            m_intensity,
            m_scratch,
            m_scratch_size ) );
    }

    if ( m_avgColor )
    {
        OPTIX_CHECK( optixDenoiserComputeAverageColor(
            m_denoiser,
            m_stream,
            &m_denoiseLayer.input,
            m_avgColor,
            m_scratch,
            m_scratch_size ) );
    }

    OPTIX_CHECK( optixUtilDenoiserInvokeTiled(
        m_denoiser,
        m_stream,
        &denoiserParams,
        m_state,
        m_state_size,
        &m_guideLayer,
        &m_denoiseLayer,
        1,
        m_scratch,
        m_scratch_size,
        m_overlap,
        m_tileWidth,
        m_tileHeight ) );
}

void OptixDenoiser_::Denoise( const OptixDenoiserImageData& data )
{
    if ( data.color == nullptr || data.output == nullptr )
    {
        return;
    }

    if ( m_temporalMode )
    {
        std::swap( m_guideLayer.outputInternalGuideLayer, m_guideLayer.previousOutputInternalGuideLayer );

        std::swap( m_outputTexture, m_previousOutputTexture );
        m_denoiseLayer.output = m_outputTexture->optixImage;
        m_denoiseLayer.previousOutput = m_firstFrame ? m_denoiseLayer.input : m_previousOutputTexture->optixImage;
    }

    // Update input textures
    std::vector<ID3D12Resource*> inputBuffers = { data.color };
    std::vector<ID3D12Resource*> internalInputBuffers = { m_colorTexture->d3dBuffer };
    if ( data.albedo && m_guideAlbedoTexture )
    {
        inputBuffers.push_back( data.albedo );
        internalInputBuffers.push_back( m_guideAlbedoTexture->d3dBuffer );
    }
    if ( data.normal && m_guideNormalTexture )
    {
        inputBuffers.push_back( data.normal );
        internalInputBuffers.push_back( m_guideNormalTexture->d3dBuffer );
    }
    if ( data.flow && m_guideFlowTexture )
    {
        inputBuffers.push_back( data.flow );
        internalInputBuffers.push_back( m_guideFlowTexture->d3dBuffer );
    }
    RHI::CopyFromUnityBuffers( inputBuffers.data(), internalInputBuffers.data(), (int)inputBuffers.size() );

    // Sync between D3D and CUDA
    RHI::D3DSignalCudaWait( m_stream, m_cudaWaitFence.get() );

    // Denoise
    DenoiseInternal();

    // Sync Between CUDA and D3D
    RHI::CudaSignalD3DWait( m_stream, m_d3dWaitFence.get() );

    // Copy output
    ID3D12Resource* outputBuffers[] = { data.output };
    ID3D12Resource* internalOutputBuffers[] = { m_outputTexture->d3dBuffer };
    RHI::CopyToUnityBuffers( internalOutputBuffers, outputBuffers, 1 );

    m_firstFrame = false;
}

}