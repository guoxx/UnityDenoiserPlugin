#include "RHI.h"
#include "Exception.h"
#include <vector>
#include <list>

extern IUnityGraphicsD3D12v7* UnityRenderAPI_D3D12();

namespace UnityDenoiserPlugin
{

static ID3D12Resource* CreateBuffer( ID3D12Device* d3dDevice, size_t sizeInBytes )
{
    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = sizeInBytes;
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_HEAP_FLAGS heapFlags = D3D12_HEAP_FLAG_SHARED;
    D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;

    ID3D12Resource* d3dResource;
    HRESULT res = d3dDevice->CreateCommittedResource(
        &heapProps,
        heapFlags,
        &resourceDesc,
        initialState,
        nullptr,
        IID_PPV_ARGS( &d3dResource ) );

    return d3dResource;
}

static D3D12_RESOURCE_BARRIER TransitionBarrier( ID3D12Resource* pResource, UINT subresource, D3D12_RESOURCE_STATES stateBefore, D3D12_RESOURCE_STATES stateAfter )
{
    D3D12_RESOURCE_BARRIER barrier;
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = pResource;
    barrier.Transition.Subresource = subresource;
    barrier.Transition.StateBefore = stateBefore;
    barrier.Transition.StateAfter = stateAfter;
    return barrier;
}

Fence::Fence( )
{
    ID3D12Device* d3d12Device = RHI::GetDevice();

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

Fence::~Fence() noexcept( false )
{
    CUDA_CHECK( cudaDestroyExternalSemaphore( cudaSempaphore ) );
    d3dFence->Release();
}

OptixTexture::OptixTexture( int w, int h, int c )
{
    const uint32_t pixelStrideInBytes = c * sizeof( float );
    const uint32_t rowStrideInBytes = pixelStrideInBytes * w;
    const uint64_t totalSizeInBytes = rowStrideInBytes * h;

    OptixPixelFormat pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    if ( c == 1 )
        pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT1;
    else if ( c == 2 )
        pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT2;
    else if ( c == 3 )
        pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT3;

    d3dBuffer = CreateBuffer( RHI::GetDevice(), totalSizeInBytes );

    HANDLE sharedHandle;
    RHI::GetDevice()->CreateSharedHandle( d3dBuffer, nullptr, GENERIC_ALL, nullptr, &sharedHandle );

    cudaExternalMemoryHandleDesc memHandleDesc = {};
    memset( &memHandleDesc, 0, sizeof( memHandleDesc ) );
    memHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    memHandleDesc.handle.win32.handle = sharedHandle;
    memHandleDesc.size = totalSizeInBytes;
    memHandleDesc.flags |= cudaExternalMemoryDedicated;
    CUDA_CHECK( cudaImportExternalMemory( &extMem, &memHandleDesc ) );

    cudaExternalMemoryBufferDesc memDesc = {};
    memDesc.offset = 0;
    memDesc.size = totalSizeInBytes;
    memDesc.flags = 0;
    CUDA_CHECK( cudaExternalMemoryGetMappedBuffer( &devPtr, extMem, &memDesc ) );
    CUDA_CHECK( cudaMemset( devPtr, 0, totalSizeInBytes ) );

    optixImage.data = (CUdeviceptr)( devPtr );
    optixImage.width = w;
    optixImage.height = h;
    optixImage.pixelStrideInBytes = pixelStrideInBytes;
    optixImage.rowStrideInBytes = rowStrideInBytes;
    optixImage.format = pixelFormat;

    CloseHandle( sharedHandle );
}

OptixTexture::~OptixTexture()
{
    cudaFree( devPtr );
    cudaDestroyExternalMemory( extMem );
    d3dBuffer->Release();
}

OIDNTexture::OIDNTexture( OIDNDevice oidnDevice, int w, int h, int c )
{
    const uint64_t totalSizeInBytes = w * h * c * sizeof( float );

    d3dBuffer = CreateBuffer( RHI::GetDevice(), totalSizeInBytes );

    HANDLE sharedHandle;
    RHI::GetDevice()->CreateSharedHandle( d3dBuffer, nullptr, GENERIC_ALL, nullptr, &sharedHandle );

    oidnBuffer = oidnNewSharedBufferFromWin32Handle( oidnDevice,
                                                     OIDNExternalMemoryTypeFlag::OIDN_EXTERNAL_MEMORY_TYPE_FLAG_D3D12_RESOURCE,
                                                     sharedHandle,
                                                     nullptr,
                                                     totalSizeInBytes );

    CloseHandle( sharedHandle );
}

OIDNTexture::~OIDNTexture()
{
    d3dBuffer->Release();
    oidnReleaseBuffer( oidnBuffer );
}

void RHI::CopyFromUnityBuffers( ID3D12Resource* fromBuffer[], ID3D12Resource* toBuffer[], int numBuffers )
{
    CommandListChunk cmdlist = GetCommandList();

    std::vector<UnityGraphicsD3D12ResourceState> fromBufferStates( numBuffers );
    std::vector<D3D12_RESOURCE_BARRIER> toBufferStatesBefore( numBuffers );
    std::vector<D3D12_RESOURCE_BARRIER> toBufferStatesAfter( numBuffers );

    for ( int i = 0; i < numBuffers; ++i )
    {
        fromBufferStates[i] = UnityGraphicsD3D12ResourceState{ fromBuffer[ i ], D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE };
        toBufferStatesBefore[i] = TransitionBarrier( toBuffer[ i ], 0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST );
        toBufferStatesAfter[i] = TransitionBarrier( toBuffer[ i ], 0, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON );
    }

    cmdlist.commandList->ResourceBarrier( (uint32_t)toBufferStatesBefore.size(), toBufferStatesBefore.data() );
    for ( int i = 0; i < numBuffers; ++i )
    {
        cmdlist.commandList->CopyResource( toBuffer[ i ], fromBuffer[ i ] );
    }
    cmdlist.commandList->ResourceBarrier( (uint32_t)toBufferStatesAfter.size(), toBufferStatesAfter.data() );

    ExecuteCommandList( cmdlist, (int)fromBufferStates.size(), fromBufferStates.data() );
}

void RHI::CopyToUnityBuffers( ID3D12Resource* fromBuffer[], ID3D12Resource* toBuffer[], int numBuffers )
{
    CommandListChunk cmdlist = GetCommandList();

    std::vector<UnityGraphicsD3D12ResourceState> toBufferStates( numBuffers );
    std::vector<D3D12_RESOURCE_BARRIER> fromBufferStatesBefore( numBuffers );
    std::vector<D3D12_RESOURCE_BARRIER> fromBufferStatesAfter( numBuffers );

    for ( int i = 0; i < numBuffers; ++i )
    {
        toBufferStates[i] = UnityGraphicsD3D12ResourceState{ toBuffer[ i ], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_DEST };
        fromBufferStatesBefore[i] = TransitionBarrier( fromBuffer[ i ], 0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE );
        fromBufferStatesAfter[i] = TransitionBarrier( fromBuffer[ i ], 0, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON );
    }

    cmdlist.commandList->ResourceBarrier( (uint32_t)fromBufferStatesBefore.size(), fromBufferStatesBefore.data() );
    for ( int i = 0; i < numBuffers; ++i )
    {
        cmdlist.commandList->CopyResource( toBuffer[ i ], fromBuffer[ i ] );
    }
    cmdlist.commandList->ResourceBarrier( (uint32_t)fromBufferStatesAfter.size(), fromBufferStatesAfter.data() );

    ExecuteCommandList( cmdlist, (int)toBufferStates.size(), toBufferStates.data() );
}

void RHI::CudaSignalD3DWait( CUstream cudaStream, Fence* fence )
{
    fence->fenceValue += 1;

    cudaExternalSemaphoreSignalParams params = {};
    memset( &params, 0, sizeof( params ) );
    params.params.fence.value = fence->fenceValue;
    cudaSignalExternalSemaphoresAsync( &fence->cudaSempaphore, &params, 1, cudaStream );

    GetCommandQueue()->Wait( fence->d3dFence, fence->fenceValue );
}

void RHI::D3DSignalCudaWait( CUstream cudaStream, Fence* fence )
{
    fence->fenceValue += 1;

    GetCommandQueue()->Signal( fence->d3dFence, fence->fenceValue );

    cudaExternalSemaphoreWaitParams params = {};
    memset( &params, 0, sizeof( params ) );
    params.params.fence.value = fence->fenceValue;
    cudaWaitExternalSemaphoresAsync( &fence->cudaSempaphore, &params, 1, cudaStream );
}

ID3D12Device* RHI::GetDevice()
{
    return UnityRenderAPI_D3D12()->GetDevice();
}

ID3D12CommandQueue* RHI::GetCommandQueue()
{
    return UnityRenderAPI_D3D12()->GetCommandQueue();
}

std::list<RHI::CommandListChunk> RHI::s_commandLists = {};

RHI::CommandListChunk RHI::GetCommandList()
{
    for ( auto it = s_commandLists.begin(); it != s_commandLists.end(); ++it )
    {
        CommandListChunk cmdlist = *it;

        uint64_t completedFenceValue = cmdlist.fence->GetCompletedValue();
        if ( cmdlist.fenceValue == completedFenceValue )
        {
            cmdlist.commandAllocator->Reset();
            cmdlist.commandList->Reset( cmdlist.commandAllocator, nullptr );

            s_commandLists.erase( it );
            return cmdlist;
        }
    }
    
    CommandListChunk cmdlist {};
    GetDevice()->CreateCommandAllocator( D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS( &cmdlist.commandAllocator ) );
    GetDevice()->CreateCommandList( 0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdlist.commandAllocator, nullptr, IID_PPV_ARGS( &cmdlist.commandList ) );
    GetDevice()->CreateFence( 0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS( &cmdlist.fence ) );
    return cmdlist;
}

void RHI::ExecuteCommandList(CommandListChunk cmdlist, int stateCount, UnityGraphicsD3D12ResourceState * states)
{
    cmdlist.commandList->Close();

    UnityRenderAPI_D3D12()->ExecuteCommandList( cmdlist.commandList, stateCount, states );

    cmdlist.fenceValue += 1;
    UnityRenderAPI_D3D12()->GetCommandQueue()->Signal( cmdlist.fence, cmdlist.fenceValue );

    s_commandLists.push_back( cmdlist );
}

}