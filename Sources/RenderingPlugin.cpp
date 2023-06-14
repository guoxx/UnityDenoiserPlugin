#include <assert.h>
#include <vector>

#include <IUnityRenderingExtensions.h>
#include <IUnityGraphicsD3D12.h>
#include <IUnityLog.h>

#include "RHI.h"
#include "OptixDenoiser.h"
#include "OIDNDenoiser.h"

static IUnityGraphics* s_Graphics = NULL;
static IUnityGraphicsD3D12v7* s_RenderAPI_D3D12 = NULL;
static IUnityLog* s_Logger = nullptr;

IUnityLog* UnityLogger()
{
    return s_Logger;
}

IUnityGraphicsD3D12v7* UnityRenderAPI_D3D12()
{
    return s_RenderAPI_D3D12;
}

enum DenoiserType : int {
    OptiX = 0,
    OIDN = 1,
};

struct DenoiserConfig {
    uint32_t imageWidth;
    uint32_t imageHeight;
    int32_t guideAlbedo;
    int32_t guideNormal;
    int32_t temporalMode;
    int32_t cleanAux;
    int32_t prefilterAux;
};

struct RenderEventData {
    intptr_t denoiseContext;
    intptr_t albedo;
    intptr_t normal;
    intptr_t flow;
    intptr_t color;
    intptr_t output;
};

static void OnRenderOptix( RenderEventData* data )
{
    UnityDenoiserPlugin::OptixDenoiserImageData imageData;
    imageData.albedo = reinterpret_cast<ID3D12Resource*>( data->albedo );
    imageData.normal = reinterpret_cast<ID3D12Resource*>( data->normal );
    imageData.flow = reinterpret_cast<ID3D12Resource*>( data->flow );
    imageData.color = reinterpret_cast<ID3D12Resource*>( data->color );
    imageData.output = reinterpret_cast<ID3D12Resource*>( data->output );

    auto ctx = reinterpret_cast<UnityDenoiserPlugin::OptixDenoiser_*>( data->denoiseContext );
    ctx->Denoise( imageData );
}

static void OnRenderOIDN( RenderEventData* data )
{
    UnityDenoiserPlugin::OIDNDenoiserImageData imageData;
    imageData.albedo = reinterpret_cast<ID3D12Resource*>( data->albedo );
    imageData.normal = reinterpret_cast<ID3D12Resource*>( data->normal );
    imageData.color = reinterpret_cast<ID3D12Resource*>( data->color );
    imageData.output = reinterpret_cast<ID3D12Resource*>( data->output );

    auto ctx = reinterpret_cast<UnityDenoiserPlugin::OIDNDenoiser*>( data->denoiseContext );
    ctx->Denoise( imageData );
}

extern "C" void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces)
{
    s_Graphics = unityInterfaces->Get<IUnityGraphics>();
    s_RenderAPI_D3D12 = unityInterfaces->Get<IUnityGraphicsD3D12v7>();
    s_Logger = unityInterfaces->Get<IUnityLog>();

    UNITY_LOG( s_Logger, __FUNCTION__ );
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
    UNITY_LOG( s_Logger, __FUNCTION__ );

    s_Graphics = nullptr;
    s_RenderAPI_D3D12 = nullptr;
    s_Logger = nullptr;
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API CreateDenoiser( DenoiserType type, const DenoiserConfig& cfg )
{
    UNITY_LOG( s_Logger, __FUNCTION__ );

    if ( type == DenoiserType::OptiX )
    {
        UnityDenoiserPlugin::OptixDenoiserConfig p = {};
        p.imageWidth = cfg.imageWidth;
        p.imageHeight = cfg.imageHeight;
        p.tileWidth = 0;
        p.tileHeight = 0;
        p.guideAlbedo = cfg.guideAlbedo;
        p.guideNormal = cfg.guideNormal;
        p.temporalMode = cfg.temporalMode;
        return reinterpret_cast<intptr_t>( new UnityDenoiserPlugin::OptixDenoiser_( p ) );
    }
    else if ( type == DenoiserType::OIDN )
    {
        UnityDenoiserPlugin::OIDNDenoiserConfig p = {};
        p.imageWidth = cfg.imageWidth;
        p.imageHeight = cfg.imageHeight;
        p.guideAlbedo = cfg.guideAlbedo;
        p.guideNormal = cfg.guideNormal;
        p.cleanAux = cfg.cleanAux;
        p.prefilterAux = cfg.prefilterAux;
        return reinterpret_cast<intptr_t>( new UnityDenoiserPlugin::OIDNDenoiser( p ) );
    }
    else
    {
        UNITY_LOG_ERROR( s_Logger, "Unknown denoiser type" );
        return 0;
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DestroyDenoiser( DenoiserType type, intptr_t ptr )
{
    UNITY_LOG( s_Logger, __FUNCTION__ );

    if ( type == DenoiserType::OptiX )
    {
        auto denoiser = reinterpret_cast<UnityDenoiserPlugin::OptixDenoiser_*>( ptr );
        delete denoiser;
    }
    else if ( type == DenoiserType::OIDN )
    {
        auto denoiser = reinterpret_cast<UnityDenoiserPlugin::OIDNDenoiser*>( ptr );
        delete denoiser;
    }
    else
    {
        UNITY_LOG_ERROR( s_Logger, "Unknown denoiser type" );
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API OnRenderEvent( int eventId, void* data )
{
    if ( eventId == DenoiserType::OptiX )
    {
        OnRenderOptix( reinterpret_cast<RenderEventData*>( data ) );
    }
    else if ( eventId == DenoiserType::OIDN )
    {
        OnRenderOIDN( reinterpret_cast<RenderEventData*>( data ) );
    }
    else
    {
        UNITY_LOG_ERROR( s_Logger, "Unknown event id" );
    }
}

// Freely defined function to pass a callback to plugin-specific scripts
extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
    return OnRenderEvent;
}

