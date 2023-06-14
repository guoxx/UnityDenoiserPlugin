#pragma once

#include <cuda_runtime.h>
#include <OpenImageDenoise/oidn.h>
#include <optix.h>
#include <list>

namespace UnityDenoiserPlugin
{

class Fence
{
public:
    uint64_t fenceValue = 0;
    ID3D12Fence* d3dFence = nullptr;
    cudaExternalSemaphore_t cudaSempaphore = nullptr;

    Fence();
    ~Fence() noexcept( false );
};

class OptixTexture
{
public:
    OptixImage2D optixImage = {};
    ID3D12Resource* d3dBuffer = nullptr;

    OptixTexture( int w, int h, int c );
    ~OptixTexture();

private:
    cudaExternalMemory_t extMem = nullptr;
    void* devPtr = nullptr;
};

class OIDNTexture
{
public:
    OIDNBuffer oidnBuffer = nullptr;
    ID3D12Resource* d3dBuffer = nullptr;

    OIDNTexture( OIDNDevice oidnDevice, int w, int h, int c );
    ~OIDNTexture();
};

class RHI
{
public:
    static void CopyFromUnityBuffers(ID3D12Resource* fromBuffer[], ID3D12Resource* toBuffer[], int numBuffers);
    static void CopyToUnityBuffers(ID3D12Resource* fromBuffer[], ID3D12Resource* toBuffer[], int numBuffers);

    static void CudaSignalD3DWait( CUstream cudaStream, Fence* fence );
    static void D3DSignalCudaWait( CUstream cudaStream, Fence* fence );

    static ID3D12Device* GetDevice();

private:
    static ID3D12CommandQueue* GetCommandQueue();

    struct CommandListChunk
    {
        ID3D12GraphicsCommandList* commandList = nullptr;
        ID3D12CommandAllocator* commandAllocator = nullptr;
        ID3D12Fence* fence = nullptr;
        uint64_t fenceValue = 0;
    };
    static std::list<CommandListChunk> s_commandLists;

    static CommandListChunk GetCommandList();
    static void ExecuteCommandList(CommandListChunk cmdlist, int stateCount, UnityGraphicsD3D12ResourceState * states);
};

}