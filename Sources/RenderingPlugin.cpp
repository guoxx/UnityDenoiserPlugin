#include <assert.h>
#include <vector>

#include <IUnityRenderingExtensions.h>
#include <IUnityGraphicsD3D12.h>
#include <IUnityLog.h>

#include "RHI.h"
#include "OptixDenoiseContext.h"
#include "OidnDenoiseContext.h"

static IUnityGraphics* s_Graphics = NULL;
static IUnityGraphicsD3D12v7* s_RenderAPI_D3D12 = NULL;
static IUnityLog* s_Logger = nullptr;
static std::string s_OidnBaseWeightsPath = "";
static std::string s_OidnPluginsFolder = "";

IUnityLog* UnityLogger()
{
    return s_Logger;
}

IUnityGraphicsD3D12v7* UnityRenderAPI_D3D12()
{
    return s_RenderAPI_D3D12;
}

enum DenoiserType : int {
    Optix = 0,
    Oidn = 1,
};

struct DenoiseConfig {
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t guideAlbedo;
    uint32_t guideNormal;
    uint32_t temporalMode;
};

struct DenoiseEventData {
    intptr_t denoiseContext;
    intptr_t albedo;
    intptr_t normal;
    intptr_t flow;
    intptr_t color;
    intptr_t output;
    uint32_t readback;
    intptr_t readbackTexture;
};

static void OnDenoiseOptix( DenoiseEventData* data )
{
    UnityDenoiserPlugin::OptixDenoiseImageData imageData;
    imageData.albedo = reinterpret_cast<ID3D12Resource*>( data->albedo );
    imageData.normal = reinterpret_cast<ID3D12Resource*>( data->normal );
    imageData.flow = reinterpret_cast<ID3D12Resource*>( data->flow );
    imageData.color = reinterpret_cast<ID3D12Resource*>( data->color );
    imageData.output = reinterpret_cast<ID3D12Resource*>( data->output );
    imageData.readback = static_cast<UnityDenoiserPlugin::Readback>( data->readback );
    imageData.readbackTexture = reinterpret_cast<ID3D12Resource*>( data->readbackTexture );

    auto ctx = reinterpret_cast<UnityDenoiserPlugin::OptixDenoiseContext*>( data->denoiseContext );
    ctx->Denoise( imageData );
}

static void OnDenoiseOdin( DenoiseEventData* data )
{
    UnityDenoiserPlugin::OidnDenoiseImageData imageData;
    imageData.albedo = reinterpret_cast<ID3D12Resource*>( data->albedo );
    imageData.normal = reinterpret_cast<ID3D12Resource*>( data->normal );
    imageData.color = reinterpret_cast<ID3D12Resource*>( data->color );
    imageData.output = reinterpret_cast<ID3D12Resource*>( data->output );
    imageData.readback = static_cast<UnityDenoiserPlugin::Readback>( data->readback );
    imageData.readbackTexture = reinterpret_cast<ID3D12Resource*>( data->readbackTexture );

    auto ctx = reinterpret_cast<UnityDenoiserPlugin::OidnDenoiseContext*>( data->denoiseContext );
    ctx->Denoise( imageData );
}

static void OnGraphicsDeviceEvent( UnityGfxDeviceEventType eventType )
{
    if ( eventType == UnityGfxDeviceEventType::kUnityGfxDeviceEventInitialize )
    {
        UnityDenoiserPlugin::RHI::Initialize();
    }
    else if ( eventType == UnityGfxDeviceEventType::kUnityGfxDeviceEventShutdown )
    {
        UnityDenoiserPlugin::RHI::Shutdown();
    }
}

extern "C" void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces)
{
    s_Graphics = unityInterfaces->Get<IUnityGraphics>();
    s_RenderAPI_D3D12 = unityInterfaces->Get<IUnityGraphicsD3D12v7>();
    s_Logger = unityInterfaces->Get<IUnityLog>();

    UNITY_LOG( s_Logger, __FUNCTION__ );

    s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

    // Run OnGraphicsDeviceEvent(initialize) manually on plugin load
    OnGraphicsDeviceEvent(UnityGfxDeviceEventType::kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
    UNITY_LOG( s_Logger, __FUNCTION__ );

    s_Graphics = nullptr;
    s_RenderAPI_D3D12 = nullptr;
    s_Logger = nullptr;

    s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API OIDNSetPluginsAndWeightsFolder( const char* pluginsFolder, const char* baseWeightFolder )
{
    s_OidnPluginsFolder = pluginsFolder;
    s_OidnBaseWeightsPath = baseWeightFolder;
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API CreateDenoiseContext( DenoiserType type, const DenoiseConfig& cfg )
{
    UNITY_LOG( s_Logger, __FUNCTION__ );

    if ( type == DenoiserType::Optix )
    {
        UnityDenoiserPlugin::OptixDenoiseConfig p = {};
        p.imageWidth = cfg.imageWidth;
        p.imageHeight = cfg.imageHeight;
        p.tileWidth = 0;
        p.tileHeight = 0;
        p.guideAlbedo = cfg.guideAlbedo;
        p.guideNormal = cfg.guideNormal;
        p.temporalMode = cfg.temporalMode;
        return reinterpret_cast<intptr_t>( new UnityDenoiserPlugin::OptixDenoiseContext( p ) );
    }
    else if ( type == DenoiserType::Oidn )
    {
        UnityDenoiserPlugin::OidnDenoiseConfig p = {};
        p.imageWidth = cfg.imageWidth;
        p.imageHeight = cfg.imageHeight;
        p.guideAlbedo = cfg.guideAlbedo;
        p.guideNormal = cfg.guideNormal;
        p.baseWeightsPath = s_OidnBaseWeightsPath;
        p.pluginsFolder = s_OidnPluginsFolder;
        return reinterpret_cast<intptr_t>( new UnityDenoiserPlugin::OidnDenoiseContext( p ) );
    }
    else
    {
        UNITY_LOG_ERROR( s_Logger, "Unknown denoiser type" );
        return 0;
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DestroyDenoiseContext( DenoiserType type, intptr_t ptr )
{
    UNITY_LOG( s_Logger, __FUNCTION__ );

    if ( type == DenoiserType::Optix )
    {
        auto ctx = reinterpret_cast<UnityDenoiserPlugin::OptixDenoiseContext*>( ptr );
        delete ctx;
    }
    else if ( type == DenoiserType::Oidn )
    {
        auto ctx = reinterpret_cast<UnityDenoiserPlugin::OidnDenoiseContext*>( ptr );
        delete ctx;
    }
    else
    {
        UNITY_LOG_ERROR( s_Logger, "Unknown denoiser type" );
    }
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API OnRenderEvent( int eventId, void* data )
{
    if ( eventId == DenoiserType::Optix )
    {
        OnDenoiseOptix( reinterpret_cast<DenoiseEventData*>( data ) );
    }
    else if ( eventId == DenoiserType::Oidn )
    {
        OnDenoiseOdin( reinterpret_cast<DenoiseEventData*>( data ) );
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

