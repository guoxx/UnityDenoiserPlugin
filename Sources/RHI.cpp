#include "RHI.h"

#include <nvrhi/d3d12.h>
#include <nvrhi/validation.h>

#include "Utils.h"
#include "Exception.h"

#include "Shaders/CopyContentBufferToTexture.h"
#include "Shaders/CopyContentTextureToBuffer.h"

extern IUnityGraphicsD3D12v7* UnityRenderAPI_D3D12();

namespace UnityDenoiserPlugin
{

static nvrhi::Format _OptixPixelFormatToNVRHIFormat( OptixPixelFormat format, uint32_t& bytesPerPixel )
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

        case OPTIX_PIXEL_FORMAT_FLOAT4:
            bytesPerPixel = sizeof( float ) * 4;
            return nvrhi::Format::RGBA32_FLOAT;

        default:
            bytesPerPixel = 0;
            return nvrhi::Format::UNKNOWN;
    }
}

static nvrhi::Format _DXGIFormatToNVRHIFormat( DXGI_FORMAT format )
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

        case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            return nvrhi::Format::RGBA32_FLOAT;

        default:
            return nvrhi::Format::UNKNOWN;
    }
}

static nvrhi::TextureHandle _CreateHandleForNativeTexture( nvrhi::DeviceHandle rhiDevice, ID3D12Resource* pTexture )
{
    auto d3dTextureDesc = pTexture->GetDesc();

    auto textureDesc = nvrhi::TextureDesc()
                           .setDimension( nvrhi::TextureDimension::Texture2D )
                           .setWidth( (uint32_t)( d3dTextureDesc.Width ) )
                           .setHeight( d3dTextureDesc.Height )
                           .setFormat( _DXGIFormatToNVRHIFormat( d3dTextureDesc.Format ) )
                           .setIsUAV( d3dTextureDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS );
    auto textureHandle = rhiDevice->createHandleForNativeTexture( nvrhi::ObjectTypes::D3D12_Resource,
                                                                  nvrhi::Object( pTexture ),
                                                                  textureDesc );
    return textureHandle;
}

GPUFence::GPUFence( nvrhi::DeviceHandle rhiDevice )
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

GPUFence::~GPUFence() noexcept( false )
{
    CUDA_CHECK( cudaDestroyExternalSemaphore( cudaSempaphore ) );
    d3dFence->Release();
}

GPUTexture::GPUTexture( nvrhi::DeviceHandle rhiDevice, int width, int height, OptixPixelFormat format )
{
    uint32_t pixelSizeInBytes = 0;
    nvrhi::Format nvrhiFormat = _OptixPixelFormatToNVRHIFormat( format, pixelSizeInBytes );

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

GPUTexture::~GPUTexture()
{
}

class NVRHIMessageCallback : public nvrhi::IMessageCallback
{
public:
    virtual void message( nvrhi::MessageSeverity severity, const char* messageText ) override
    {
        if ( severity == nvrhi::MessageSeverity::Info )
            LogMessage( messageText );
        else if ( severity == nvrhi::MessageSeverity::Warning )
            LogWarning( messageText );
        else
            LogError( messageText );
    }
};

namespace
{
static bool s_initialized = false;
static nvrhi::DeviceHandle s_RHIDevice = nullptr;
static nvrhi::CommandListHandle s_RHICommandList = nullptr;
static nvrhi::ComputePipelineHandle s_pipelineCopyTextureToBuffer = nullptr;
static nvrhi::ComputePipelineHandle s_pipelineCopyBufferToTexture = nullptr;
static NVRHIMessageCallback s_messageCallback;
}  // namespace

void RHI::Initialize()
{
    if ( s_initialized ) { return; }
    s_initialized = true;

    // Initialize NVRHI device
    {
        nvrhi::d3d12::DeviceDesc deviceDesc = {};
        deviceDesc.errorCB = &s_messageCallback;
        deviceDesc.pDevice = UnityRenderAPI_D3D12()->GetDevice();
        deviceDesc.pGraphicsCommandQueue = UnityRenderAPI_D3D12()->GetCommandQueue();
        deviceDesc.pComputeCommandQueue = UnityRenderAPI_D3D12()->GetCommandQueue();
        deviceDesc.pCopyCommandQueue = UnityRenderAPI_D3D12()->GetCommandQueue();
        s_RHIDevice = nvrhi::d3d12::createDevice( deviceDesc );

#if defined( DEBUG ) || defined( _DEBUG )
        nvrhi::DeviceHandle nvrhiValidationLayer = nvrhi::validation::createValidationLayer( s_RHIDevice );
        s_RHIDevice = nvrhiValidationLayer;
#endif
    }

    // Initialize NVRHI command lsit
    s_RHICommandList = s_RHIDevice->createCommandList( nvrhi::CommandListParameters{} );

    // Initialize pipelines
    {
        nvrhi::ShaderHandle computeShader = s_RHIDevice->createShader(
            nvrhi::ShaderDesc( nvrhi::ShaderType::Compute ),
            (const void*)g_CopyContentTextureToBuffer_ByteCode,
            sizeof( g_CopyContentTextureToBuffer_ByteCode ) );

        auto layoutDesc = nvrhi::BindingLayoutDesc()
                              .setVisibility( nvrhi::ShaderType::Compute )
                              .setRegisterSpace( 0 )
                              .addItem( nvrhi::BindingLayoutItem::Texture_SRV( 0 ) )
                              .addItem( nvrhi::BindingLayoutItem::TypedBuffer_UAV( 0 ) )
                              .addItem( nvrhi::BindingLayoutItem::PushConstants( 0, sizeof( uint4 ) ) );
        auto bindingLayout = s_RHIDevice->createBindingLayout( layoutDesc );

        auto pipelineDesc = nvrhi::ComputePipelineDesc()
                                .setComputeShader( computeShader )
                                .addBindingLayout( bindingLayout );
        s_pipelineCopyTextureToBuffer = s_RHIDevice->createComputePipeline( pipelineDesc );
    }

    {
        nvrhi::ShaderHandle computeShader = s_RHIDevice->createShader(
            nvrhi::ShaderDesc( nvrhi::ShaderType::Compute ),
            (const void*)g_CopyContentBufferToTexture_ByteCode,
            sizeof( g_CopyContentBufferToTexture_ByteCode ) );

        auto layoutDesc = nvrhi::BindingLayoutDesc()
                              .setVisibility( nvrhi::ShaderType::Compute )
                              .setRegisterSpace( 0 )
                              .addItem( nvrhi::BindingLayoutItem::Texture_UAV( 0 ) )
                              .addItem( nvrhi::BindingLayoutItem::TypedBuffer_SRV( 0 ) )
                              .addItem( nvrhi::BindingLayoutItem::PushConstants( 0, sizeof( uint4 ) ) );
        auto bindingLayout = s_RHIDevice->createBindingLayout( layoutDesc );

        auto pipelineDesc = nvrhi::ComputePipelineDesc()
                                .setComputeShader( computeShader )
                                .addBindingLayout( bindingLayout );
        s_pipelineCopyBufferToTexture = s_RHIDevice->createComputePipeline( pipelineDesc );
    }
}

void RHI::Shutdown()
{
    if ( !s_initialized ) { return; }
    s_initialized = false;

    s_RHICommandList = nullptr;
    s_RHIDevice = nullptr;
}

nvrhi::DeviceHandle RHI::GetD3D12Device()
{
    return s_RHIDevice;
}

nvrhi::CommandListHandle RHI::GetD3D12CommandList()
{
    return s_RHICommandList;
}

void RHI::SignalD3D12Fence(ID3D12Fence* fence, uint64_t value)
{
    ID3D12CommandQueue* d3dQueue = s_RHIDevice->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Graphics);
    d3dQueue->Signal( fence, value );
}

void RHI::WaitD3D12Fence(ID3D12Fence* fence, uint64_t value)
{
    ID3D12CommandQueue* d3dQueue = s_RHIDevice->getNativeQueue(nvrhi::ObjectTypes::D3D12_CommandQueue, nvrhi::CommandQueue::Graphics);
    d3dQueue->Wait( fence, value );
}

void RHI::CopyContent( nvrhi::TextureHandle fromTexture, nvrhi::BufferHandle toBuffer )
{
    s_RHICommandList->open();

    nvrhi::BindingSetHandle bindingSet;
    {
        auto bindingSetDesc = nvrhi::BindingSetDesc()
                              .addItem( nvrhi::BindingSetItem::Texture_SRV( 0, fromTexture ) )
                              .addItem( nvrhi::BindingSetItem::TypedBuffer_UAV( 0, toBuffer ) )
                              .addItem( nvrhi::BindingSetItem::PushConstants( 0, sizeof( uint4 ) ) );

        auto pipelineDesc = s_pipelineCopyTextureToBuffer->getDesc();

        bindingSet = s_RHIDevice->createBindingSet( bindingSetDesc, pipelineDesc.bindingLayouts.front() );
    }

    nvrhi::ComputeState computeState = {};
    computeState.setPipeline( s_pipelineCopyTextureToBuffer );
    computeState.addBindingSet( bindingSet );
    s_RHICommandList->setComputeState( computeState );

    uint4 constants;
    s_RHICommandList->setPushConstants( &constants, sizeof(constants) );

    const uint32_t threadSize = 8;
    const uint32_t groupSizeX = DivideRoundUp( fromTexture->getDesc().width, threadSize ) * threadSize;
    const uint32_t groupSizeY = DivideRoundUp( fromTexture->getDesc().height, threadSize ) * threadSize;
    s_RHICommandList->dispatch( groupSizeX, groupSizeY, 1 );

    s_RHICommandList->close();
    s_RHIDevice->executeCommandList( s_RHICommandList );
}

void RHI::CopyContent( nvrhi::BufferHandle fromBuffer, nvrhi::TextureHandle toTexture )
{
    s_RHICommandList->open();

    nvrhi::BindingSetHandle bindingSet;
    {
        auto bindingSetDesc = nvrhi::BindingSetDesc()
                              .addItem( nvrhi::BindingSetItem::Texture_UAV( 0, toTexture ) )
                              .addItem( nvrhi::BindingSetItem::TypedBuffer_SRV( 0, fromBuffer ) )
                              .addItem( nvrhi::BindingSetItem::PushConstants( 0, sizeof( uint4 ) ) );

        auto pipelineDesc = s_pipelineCopyBufferToTexture->getDesc();

        bindingSet = s_RHIDevice->createBindingSet( bindingSetDesc, pipelineDesc.bindingLayouts.front() );
    }

    nvrhi::ComputeState computeState = {};
    computeState.setPipeline( s_pipelineCopyBufferToTexture );
    computeState.addBindingSet( bindingSet );
    s_RHICommandList->setComputeState( computeState );

    uint4 constants;
    s_RHICommandList->setPushConstants( &constants, sizeof( constants ) );

    const uint32_t threadSize = 8;
    const uint32_t groupSizeX = DivideRoundUp( toTexture->getDesc().width, threadSize ) * threadSize;
    const uint32_t groupSizeY = DivideRoundUp( toTexture->getDesc().height, threadSize ) * threadSize;
    s_RHICommandList->dispatch( groupSizeX, groupSizeY, 1 );

    s_RHICommandList->close();
    s_RHIDevice->executeCommandList( s_RHICommandList );
}

void RHI::CopyContent( ID3D12Resource* fromTexture, nvrhi::BufferHandle toBuffer )
{
    nvrhi::TextureHandle textureHandle = _CreateHandleForNativeTexture( s_RHIDevice, fromTexture );
    CopyContent( textureHandle, toBuffer );
}

void RHI::CopyContent( nvrhi::BufferHandle fromBuffer, ID3D12Resource* toTexture )
{
    nvrhi::TextureHandle textureHandle = _CreateHandleForNativeTexture( s_RHIDevice, toTexture );
    CopyContent( fromBuffer, textureHandle );
}

}