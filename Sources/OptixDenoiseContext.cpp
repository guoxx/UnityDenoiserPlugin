#include "OptixDenoiseContext.h"
#include "Exception.h"

#include <optix.h>
#include <optix_denoiser_tiling.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <nvrhi/nvrhi.h>
#include <nvrhi/d3d12.h>
#include <nvrhi/validation.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

#include "Shaders/CopyContentBufferToTexture.h"
#include "Shaders/CopyContentTextureToBuffer.h"


extern IUnityGraphicsD3D12v7* UnityRenderAPI_D3D12();
extern IUnityLog* UnityLogger();
#define UNITY_LOG_(MSG_) UNITY_LOG(UnityLogger(), MSG_)
#define UNITY_LOG_WARNING_(MSG_) UNITY_LOG_WARNING(UnityLogger(), MSG_)
#define UNITY_LOG_ERROR_(MSG_) UNITY_LOG_ERROR(UnityLogger(), MSG_)


namespace UnityDenoisePlugin
{

// Utility functions
static nvrhi::Format OptixPixelFormatToNVRHIFormat( OptixPixelFormat format, uint32_t& bytesPerPixel )
{
    switch ( format ) {
        case OPTIX_PIXEL_FORMAT_UCHAR4:
            bytesPerPixel = sizeof( uint8_t ) * 4;
            return nvrhi::Format::RGBA8_UNORM;

        case OPTIX_PIXEL_FORMAT_HALF2:
            bytesPerPixel = sizeof( uint16_t ) * 2;
            return nvrhi::Format::RG16_FLOAT;

        case OPTIX_PIXEL_FORMAT_HALF4:
            bytesPerPixel = sizeof( uint16_t ) * 4;
            return nvrhi::Format::RGBA16_FLOAT;

        case OPTIX_PIXEL_FORMAT_FLOAT2:
            bytesPerPixel = sizeof( float ) * 2;
            return nvrhi::Format::RG32_FLOAT;

        case OPTIX_PIXEL_FORMAT_FLOAT3:
            bytesPerPixel = sizeof( float ) * 3;
            return nvrhi::Format::RGB32_FLOAT;

        case OPTIX_PIXEL_FORMAT_FLOAT4:
            bytesPerPixel = sizeof( float ) * 4;
            return nvrhi::Format::RGBA32_FLOAT;

        default:
            bytesPerPixel = 0;
            return nvrhi::Format::UNKNOWN;
    }
}

static nvrhi::Format DXGIFormatToNVRHIFormat( DXGI_FORMAT format )
{
    switch ( format ) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            return nvrhi::Format::RGBA8_UNORM;

        case DXGI_FORMAT_R16G16_TYPELESS:
        case DXGI_FORMAT_R16G16_FLOAT:
            return nvrhi::Format::RG16_FLOAT;

        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return nvrhi::Format::RGBA16_FLOAT;

        case DXGI_FORMAT_R32G32_TYPELESS:
        case DXGI_FORMAT_R32G32_FLOAT:
            return nvrhi::Format::RG32_FLOAT;

        case DXGI_FORMAT_R32G32B32_TYPELESS:
        case DXGI_FORMAT_R32G32B32_FLOAT:
            return nvrhi::Format::RGB32_FLOAT;

        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return nvrhi::Format::RGBA32_FLOAT;

        default:
            return nvrhi::Format::UNKNOWN;
    }
}

uint32_t DivideRoundUp( uint32_t a, uint32_t b )
{
    return ( a + b - 1 ) / b;
}

void LogCallback( uint32_t level, const char* tag, const char* message, void* /*cbdata*/ )
{
    std::ostringstream oss;
    oss << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";

    if ( level == 4 ) {
        UNITY_LOG_( oss.str().c_str() );
    } else if ( level == 3 ) {
        UNITY_LOG_WARNING_( oss.str().c_str() );
    } else {
        UNITY_LOG_ERROR_( oss.str().c_str() );
    }
}

void SignalExternalSemaphore( cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream )
{
    cudaExternalSemaphoreSignalParams params = {};
    memset( &params, 0, sizeof( params ) );
    params.params.fence.value = value;

    cudaSignalExternalSemaphoresAsync( &extSem, &params, 1, stream );
}

void WaitExternalSemaphore( cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream )
{
    cudaExternalSemaphoreWaitParams params = {};
    memset( &params, 0, sizeof( params ) );
    params.params.fence.value = value;

    cudaWaitExternalSemaphoresAsync( &extSem, &params, 1, stream );
}

nvrhi::TextureHandle CreateHandleForNativeTexture( nvrhi::DeviceHandle rhiDevice, ID3D12Resource* pTexture )
{
    auto d3dTextureDesc = pTexture->GetDesc();

    auto textureDesc = nvrhi::TextureDesc()
                           .setDimension( nvrhi::TextureDimension::Texture2D )
                           .setWidth( (uint32_t)( d3dTextureDesc.Width ) )
                           .setHeight( d3dTextureDesc.Height )
                           .setFormat( DXGIFormatToNVRHIFormat( d3dTextureDesc.Format ) )
                           .setIsUAV( d3dTextureDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS );
    auto textureHandle = rhiDevice->createHandleForNativeTexture( nvrhi::ObjectTypes::D3D12_Resource,
                                                                  nvrhi::Object( pTexture ),
                                                                  textureDesc );
    return textureHandle;
}


class GPUFence
{
public:
    ID3D12Fence* d3dFence = nullptr;
    cudaExternalSemaphore_t cudaSempaphore = nullptr;

    GPUFence( nvrhi::DeviceHandle rhiDevice )
    {
        ID3D12Device* d3d12Device = rhiDevice->getNativeObject( nvrhi::ObjectTypes::D3D12_Device );

        d3d12Device->CreateFence( 0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS( &d3dFence ) );

        HANDLE sharedHandle;
        d3d12Device->CreateSharedHandle( d3dFence, nullptr, GENERIC_ALL, nullptr, &sharedHandle );

        cudaExternalSemaphoreHandleDesc desc = {};
        memset( &desc, 0, sizeof( desc ) );
        desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
        desc.handle.win32.handle = sharedHandle;
        CUDA_CHECK( cudaImportExternalSemaphore( &cudaSempaphore, &desc ) );

        CloseHandle( sharedHandle );
    }

    ~GPUFence() noexcept( false )
    {
        CUDA_CHECK( cudaDestroyExternalSemaphore( cudaSempaphore ) );
        d3dFence->Release();
    }
};


class GPUTexture
{
public:
    OptixImage2D optixImage = {};
    nvrhi::BufferHandle bufferHandle;

    GPUTexture( nvrhi::DeviceHandle rhiDevice, int width, int height, OptixPixelFormat format )
    {
        uint32_t pixelSizeInBytes = 0;
        nvrhi::Format nvrhiFormat = OptixPixelFormatToNVRHIFormat( format, pixelSizeInBytes );

        const uint64_t totalSizeInBytes = height * width * pixelSizeInBytes;

        bufferHandle = nullptr;
        {
            auto bufferDesc = nvrhi::BufferDesc()
                                  .setByteSize( totalSizeInBytes )
                                  .setStructStride( pixelSizeInBytes )
                                  .setFormat( nvrhiFormat )
                                  .setCanHaveTypedViews( true )
                                  .setCanHaveUAVs( true )
                                  .setInitialState( nvrhi::ResourceStates::Common )
                                  .setKeepInitialState( true );
            bufferDesc.sharedResourceFlags = nvrhi::SharedResourceFlags::Shared;
            bufferHandle = rhiDevice->createBuffer( bufferDesc );
        }

        cudaExternalMemory_t extMem = NULL;
        {
            cudaExternalMemoryHandleDesc memHandleDesc = {};
            memset( &memHandleDesc, 0, sizeof( memHandleDesc ) );
            memHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
            memHandleDesc.handle.win32.handle = (void*)bufferHandle->getNativeObject( nvrhi::ObjectTypes::SharedHandle );
            memHandleDesc.size = totalSizeInBytes;
            memHandleDesc.flags |= cudaExternalMemoryDedicated;
            CUDA_CHECK( cudaImportExternalMemory( &extMem, &memHandleDesc ) );
        }

        void* ptr = nullptr;
        {
            cudaExternalMemoryBufferDesc memDesc = {};
            memDesc.offset = 0;
            memDesc.size = totalSizeInBytes;
            memDesc.flags = 0;
            CUDA_CHECK( cudaExternalMemoryGetMappedBuffer( &ptr, extMem, &memDesc ) );
            CUDA_CHECK( cudaMemset( ptr, 0, totalSizeInBytes ) );
        }

        optixImage.data = (CUdeviceptr)( ptr );
        optixImage.width = width;
        optixImage.height = height;
        optixImage.pixelStrideInBytes = pixelSizeInBytes;
        optixImage.rowStrideInBytes = width * pixelSizeInBytes;
        optixImage.format = format;
    }

    ~GPUTexture()
    {
    }
};


namespace
{
    // Static global variables
    static bool s_initialized = false;

    static nvrhi::DeviceHandle s_nvrhiDevice = nullptr;
    static nvrhi::CommandListHandle s_nvrhiCommandList = nullptr;
    static nvrhi::ComputePipelineHandle s_pipelineCopyTextureToBuffer = nullptr;
    static nvrhi::ComputePipelineHandle s_pipelineCopyBufferToTexture = nullptr;

    static OptixDeviceContext s_optixDeviceContext = nullptr;
};

static void Initialize_()
{
    if (s_initialized) { return; }
    s_initialized = true;

    // Initialize NVRHI device
    {
        nvrhi::d3d12::DeviceDesc deviceDesc = {};
        deviceDesc.pDevice = UnityRenderAPI_D3D12()->GetDevice();
        deviceDesc.pGraphicsCommandQueue = UnityRenderAPI_D3D12()->GetCommandQueue();
        deviceDesc.pComputeCommandQueue = UnityRenderAPI_D3D12()->GetCommandQueue();
        deviceDesc.pCopyCommandQueue = UnityRenderAPI_D3D12()->GetCommandQueue();
        s_nvrhiDevice = nvrhi::d3d12::createDevice( deviceDesc );

        #if defined( DEBUG ) || defined( _DEBUG )
            nvrhi::DeviceHandle nvrhiValidationLayer = nvrhi::validation::createValidationLayer( s_nvrhiDevice );
            s_nvrhiDevice = nvrhiValidationLayer;
        #endif
    }

    // Initialize NVRHI command lsit
    {
        s_nvrhiCommandList = s_nvrhiDevice->createCommandList( nvrhi::CommandListParameters{} );
    }

    // Initialize pipelines
    {
        nvrhi::ShaderHandle computeShader = s_nvrhiDevice->createShader(
            nvrhi::ShaderDesc( nvrhi::ShaderType::Compute ),
            (const void*)g_CopyContentTextureToBuffer_ByteCode,
            sizeof( g_CopyContentTextureToBuffer_ByteCode ) );

        auto layoutDesc = nvrhi::BindingLayoutDesc()
                              .setVisibility( nvrhi::ShaderType::Compute )
                              .setRegisterSpace( 0 )
                              .addItem( nvrhi::BindingLayoutItem::Texture_SRV( 0 ) )
                              .addItem( nvrhi::BindingLayoutItem::TypedBuffer_UAV( 0 ) )
                              .addItem( nvrhi::BindingLayoutItem::PushConstants( 0, sizeof( uint4 ) ) );
        auto bindingLayout = s_nvrhiDevice->createBindingLayout( layoutDesc );

        auto pipelineDesc = nvrhi::ComputePipelineDesc()
                                .setComputeShader( computeShader )
                                .addBindingLayout( bindingLayout );
        s_pipelineCopyTextureToBuffer = s_nvrhiDevice->createComputePipeline( pipelineDesc );
    }

    {
        nvrhi::ShaderHandle computeShader = s_nvrhiDevice->createShader(
            nvrhi::ShaderDesc( nvrhi::ShaderType::Compute ),
            (const void*)g_CopyContentBufferToTexture_ByteCode,
            sizeof( g_CopyContentBufferToTexture_ByteCode ) );

        auto layoutDesc = nvrhi::BindingLayoutDesc()
                              .setVisibility( nvrhi::ShaderType::Compute )
                              .setRegisterSpace( 0 )
                              .addItem( nvrhi::BindingLayoutItem::Texture_UAV( 0 ) )
                              .addItem( nvrhi::BindingLayoutItem::TypedBuffer_SRV( 0 ) )
                              .addItem( nvrhi::BindingLayoutItem::PushConstants( 0, sizeof( uint4 ) ) );
        auto bindingLayout = s_nvrhiDevice->createBindingLayout( layoutDesc );

        auto pipelineDesc = nvrhi::ComputePipelineDesc()
                                .setComputeShader( computeShader )
                                .addBindingLayout( bindingLayout );
        s_pipelineCopyBufferToTexture = s_nvrhiDevice->createComputePipeline( pipelineDesc );
    }

    // Initialize OptiX device context
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
}

static void Finalize_()
{
    if (!s_initialized) { return; }
    s_initialized = false;

    s_pipelineCopyBufferToTexture = nullptr;
    s_pipelineCopyTextureToBuffer = nullptr;
    s_nvrhiCommandList = nullptr;
    s_nvrhiDevice = nullptr;

    cudaDeviceSynchronize();

    OPTIX_CHECK( optixDeviceContextDestroy( s_optixDeviceContext ) );
    s_optixDeviceContext = nullptr;
}

void CopyContent( nvrhi::TextureHandle fromTexture, nvrhi::BufferHandle toBuffer )
{
    s_nvrhiCommandList->open();

    nvrhi::BindingSetHandle bindingSet;
    {
        auto bindingSetDesc = nvrhi::BindingSetDesc()
                              .addItem( nvrhi::BindingSetItem::Texture_SRV( 0, fromTexture ) )
                              .addItem( nvrhi::BindingSetItem::TypedBuffer_UAV( 0, toBuffer ) )
                              .addItem( nvrhi::BindingSetItem::PushConstants( 0, sizeof( uint4 ) ) );

        auto pipelineDesc = s_pipelineCopyTextureToBuffer->getDesc();

        bindingSet = s_nvrhiDevice->createBindingSet( bindingSetDesc, pipelineDesc.bindingLayouts.front() );
    }

    nvrhi::ComputeState computeState = {};
    computeState.setPipeline( s_pipelineCopyTextureToBuffer );
    computeState.addBindingSet( bindingSet );
    s_nvrhiCommandList->setComputeState( computeState );

    uint4 constants;
    s_nvrhiCommandList->setPushConstants( &constants, sizeof(constants) );

    const uint32_t threadSize = 8;
    const uint32_t groupSizeX = DivideRoundUp( fromTexture->getDesc().width, threadSize ) * threadSize;
    const uint32_t groupSizeY = DivideRoundUp( fromTexture->getDesc().height, threadSize ) * threadSize;
    s_nvrhiCommandList->dispatch( groupSizeX, groupSizeY, 1 );

    s_nvrhiCommandList->close();
    s_nvrhiDevice->executeCommandList( s_nvrhiCommandList );
}

void CopyContent( nvrhi::BufferHandle fromBuffer, nvrhi::TextureHandle toTexture )
{
    s_nvrhiCommandList->open();

    nvrhi::BindingSetHandle bindingSet;
    {
        auto bindingSetDesc = nvrhi::BindingSetDesc()
                              .addItem( nvrhi::BindingSetItem::Texture_UAV( 0, toTexture ) )
                              .addItem( nvrhi::BindingSetItem::TypedBuffer_SRV( 0, fromBuffer ) )
                              .addItem( nvrhi::BindingSetItem::PushConstants( 0, sizeof( uint4 ) ) );

        auto pipelineDesc = s_pipelineCopyBufferToTexture->getDesc();

        bindingSet = s_nvrhiDevice->createBindingSet( bindingSetDesc, pipelineDesc.bindingLayouts.front() );
    }

    nvrhi::ComputeState computeState = {};
    computeState.setPipeline( s_pipelineCopyBufferToTexture );
    computeState.addBindingSet( bindingSet );
    s_nvrhiCommandList->setComputeState( computeState );

    uint4 constants;
    s_nvrhiCommandList->setPushConstants( &constants, sizeof( constants ) );

    const uint32_t threadSize = 8;
    const uint32_t groupSizeX = DivideRoundUp( toTexture->getDesc().width, threadSize ) * threadSize;
    const uint32_t groupSizeY = DivideRoundUp( toTexture->getDesc().height, threadSize ) * threadSize;
    s_nvrhiCommandList->dispatch( groupSizeX, groupSizeY, 1 );

    s_nvrhiCommandList->close();
    s_nvrhiDevice->executeCommandList( s_nvrhiCommandList );
}

void CopyContent( ID3D12Resource* fromTexture, nvrhi::BufferHandle toBuffer )
{
    nvrhi::TextureHandle textureHandle = CreateHandleForNativeTexture( s_nvrhiDevice, fromTexture );
    CopyContent( textureHandle, toBuffer );
}

void CopyContent( nvrhi::BufferHandle fromBuffer, ID3D12Resource* toTexture )
{
    nvrhi::TextureHandle textureHandle = CreateHandleForNativeTexture( s_nvrhiDevice, toTexture );
    CopyContent( fromBuffer, textureHandle );
}

OptixDenoiseContext::OptixDenoiseContext( const OptixDenoiseConfig& cfg )
{
    Initialize_();

    // Sanity check
    SUTIL_ASSERT_MSG( !cfg.guideNormal || cfg.guideAlbedo,
                      "Currently albedo is required if normal input is given" );
    SUTIL_ASSERT_MSG( ( cfg.tileWidth == 0 && cfg.tileHeight == 0 ) || ( cfg.tileWidth > 0 && cfg.tileHeight > 0 ),
                      "tile size must be > 0 for width and height" );

    m_temporalMode = cfg.temporalMode;
    m_tileWidth = cfg.tileWidth > 0 ? cfg.tileWidth : cfg.imageWidth;
    m_tileHeight = cfg.tileHeight > 0 ? cfg.tileHeight : cfg.imageHeight;

    bool kpMode = true;

    // Initialize D3D resources
    {
        m_cudaWaitFence = std::make_shared<GPUFence>( s_nvrhiDevice );
        m_d3dWaitFence = std::make_shared<GPUFence>( s_nvrhiDevice );
    }

    // Create denoiser
    {
        OptixDenoiserOptions options = {};
        options.guideAlbedo = cfg.guideAlbedo ? 1 : 0;
        options.guideNormal = cfg.guideNormal ? 1 : 0;

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
            m_colorTexture = std::make_shared<GPUTexture>( s_nvrhiDevice, cfg.imageWidth, cfg.imageHeight, OPTIX_PIXEL_FORMAT_HALF4 );
            m_outputTexture = std::make_shared<GPUTexture>( s_nvrhiDevice, cfg.imageWidth, cfg.imageHeight, OPTIX_PIXEL_FORMAT_HALF4 );

            m_denoiseLayer = {};
            m_denoiseLayer.type = OPTIX_DENOISER_AOV_TYPE_NONE;
            m_denoiseLayer.input = m_colorTexture->optixImage;
            m_denoiseLayer.output = m_outputTexture->optixImage;

            if ( cfg.temporalMode ) {
                m_previousOutputTexture = std::make_shared<GPUTexture>( s_nvrhiDevice, cfg.imageWidth, cfg.imageHeight, OPTIX_PIXEL_FORMAT_HALF4 );
                m_denoiseLayer.previousOutput = m_previousOutputTexture->optixImage; 
            }
        }

        // Create guide layer resources
        {
            m_guideLayer = {};

            if ( cfg.guideAlbedo ) {
                m_guideAlbedoTexture = std::make_shared<GPUTexture>( s_nvrhiDevice, cfg.imageWidth, cfg.imageHeight, OPTIX_PIXEL_FORMAT_HALF4 );
                m_guideLayer.albedo = m_guideAlbedoTexture->optixImage;
            }

            if ( cfg.guideNormal ) {
                m_guideNormalTexture = std::make_shared<GPUTexture>( s_nvrhiDevice, cfg.imageWidth, cfg.imageHeight, OPTIX_PIXEL_FORMAT_HALF4 );
                m_guideLayer.normal = m_guideNormalTexture->optixImage;
            }

            if ( cfg.temporalMode ) {
                m_guideFlowTexture = std::make_shared<GPUTexture>( s_nvrhiDevice, cfg.imageWidth, cfg.imageHeight, OPTIX_PIXEL_FORMAT_HALF2 );
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

OptixDenoiseContext::~OptixDenoiseContext() noexcept( false )
{
    cudaDeviceSynchronize();

    OPTIX_CHECK( optixDenoiserDestroy( m_denoiser ) );

    CUDA_CHECK( cudaStreamDestroy( m_stream ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_intensity ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_avgColor ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_scratch ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_state ) ) );
}

void OptixDenoiseContext::DenoiseInternal( const OptixDenoiseImageData& data )
{
    OptixDenoiserParams denoiserParams = {};
    denoiserParams.denoiseAlpha = OptixDenoiserAlphaMode::OPTIX_DENOISER_ALPHA_MODE_COPY;
    denoiserParams.hdrIntensity = m_intensity;
    denoiserParams.hdrAverageColor = m_avgColor;
    denoiserParams.blendFactor = 0.0f;
    denoiserParams.temporalModeUsePreviousLayers = m_firstFrame ? 0 : 1; 

    if ( m_intensity ) {
        OPTIX_CHECK( optixDenoiserComputeIntensity(
            m_denoiser,
            m_stream,
            &m_denoiseLayer.input,
            m_intensity,
            m_scratch,
            m_scratch_size ) );
    }

    if ( m_avgColor ) {
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

void OptixDenoiseContext::Denoise( const OptixDenoiseImageData& data )
{
    if ( data.color == nullptr || data.output == nullptr ) {
        return;
    }

    m_renderSyncCounter += 1;

    if ( m_temporalMode ) {
        std::swap( m_guideLayer.outputInternalGuideLayer, m_guideLayer.previousOutputInternalGuideLayer );

        std::swap( m_outputTexture, m_previousOutputTexture );
        m_denoiseLayer.output = m_outputTexture->optixImage;
        m_denoiseLayer.previousOutput = m_firstFrame ? m_denoiseLayer.input : m_previousOutputTexture->optixImage;
    }

    CopyContent( data.color, m_colorTexture->bufferHandle );

    if ( data.albedo ) {
        CopyContent( data.albedo, m_guideAlbedoTexture->bufferHandle );
    }

    if ( data.normal ) {
        CopyContent( data.normal, m_guideNormalTexture->bufferHandle );
    }

    if ( data.flow ) {
        CopyContent( data.flow, m_guideFlowTexture->bufferHandle );
    }

    ID3D12CommandQueue* d3dQueue = s_nvrhiDevice->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Graphics);

    d3dQueue->Signal( m_cudaWaitFence->d3dFence, m_renderSyncCounter );
    WaitExternalSemaphore( m_cudaWaitFence->cudaSempaphore, m_renderSyncCounter, m_stream );

    DenoiseInternal( data );
    // CUDA_SYNC_CHECK();

    SignalExternalSemaphore( m_d3dWaitFence->cudaSempaphore, m_renderSyncCounter, m_stream );
    d3dQueue->Wait( m_d3dWaitFence->d3dFence, m_renderSyncCounter );

    CopyContent( m_outputTexture->bufferHandle, data.output );

    if ( data.readback != OptixDenoiseReadback::None && data.readbackTexture != nullptr ) {
        std::shared_ptr<GPUTexture> readbackTexture{};
        if ( data.readback == OptixDenoiseReadback::Albedo ) {
            readbackTexture = m_guideAlbedoTexture;
        } else if ( data.readback == OptixDenoiseReadback::Normal ) {
            readbackTexture = m_guideNormalTexture;
        } else if ( data.readback == OptixDenoiseReadback::Flow ) {
            readbackTexture = m_temporalMode ? m_guideFlowTexture : nullptr;
        } else if ( data.readback == OptixDenoiseReadback::Color ) {
            readbackTexture = m_colorTexture;
        } else if ( data.readback == OptixDenoiseReadback::PreviousOutput ) {
            readbackTexture = m_previousOutputTexture;
        }

        if (readbackTexture)
            CopyContent( readbackTexture->bufferHandle, data.readbackTexture );
    }

    m_firstFrame = false;
}

}