#pragma once

#include <nvrhi/nvrhi.h>
#include <cuda_runtime.h>
#include <optix.h>

namespace UnityDenoiserPlugin
{

class GPUFence
{
public:
    ID3D12Fence* d3dFence = nullptr;
    cudaExternalSemaphore_t cudaSempaphore = nullptr;

    GPUFence( nvrhi::DeviceHandle rhiDevice );
    ~GPUFence() noexcept( false );
};

class GPUTexture
{
public:
    OptixImage2D optixImage = {};
    nvrhi::BufferHandle bufferHandle;

    GPUTexture( nvrhi::DeviceHandle rhiDevice, int width, int height, OptixPixelFormat format );
    ~GPUTexture();
};

class RHI
{
public:
    static void Initialize();
    static void Shutdown();

    static nvrhi::DeviceHandle GetD3D12Device();
    static nvrhi::CommandListHandle GetD3D12CommandList();

    static void SignalD3D12Fence(ID3D12Fence* fence, uint64_t value);
    static void WaitD3D12Fence(ID3D12Fence* fence, uint64_t value);

    static void CopyContent( nvrhi::TextureHandle fromTexture, nvrhi::BufferHandle toBuffer );
    static void CopyContent( ID3D12Resource* fromTexture, nvrhi::BufferHandle toBuffer );

    static void CopyContent( nvrhi::BufferHandle fromBuffer, nvrhi::TextureHandle toTexture );
    static void CopyContent( nvrhi::BufferHandle fromBuffer, ID3D12Resource* toTexture );
};

}