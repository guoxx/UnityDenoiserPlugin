#include <assert.h>
#include <vector>

#include <IUnityRenderingExtensions.h>
#include <IUnityGraphicsD3D12.h>
#include <IUnityLog.h>

#include "OptixDenoiseContext.h"

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

extern "C" void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces)
{
    s_Graphics = unityInterfaces->Get<IUnityGraphics>();
    s_RenderAPI_D3D12 = unityInterfaces->Get<IUnityGraphicsD3D12v7>();
    s_Logger = unityInterfaces->Get<IUnityLog>();

    UNITY_LOG( s_Logger, "Load UnityDenoiserPlugin");
    //s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

    //// Run OnGraphicsDeviceEvent(initialize) manually on plugin load
    //OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
    UNITY_LOG( s_Logger,"Unload UnityDenoiserPlugin");

    s_Graphics = nullptr;
    s_RenderAPI_D3D12 = nullptr;
    s_Logger = nullptr;

    //s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
}

extern "C" intptr_t UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API CreateOptixDenoiseContext( const UnityDenoisePlugin::OptixDenoiseConfig& cfg )
{
    UNITY_LOG( s_Logger, "CreateOptixDenoiseContext" );
    return reinterpret_cast<intptr_t>( new UnityDenoisePlugin::OptixDenoiseContext( cfg ) );
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DestroyOptixDenoiseContext( intptr_t ptr )
{
    auto ctx = reinterpret_cast<UnityDenoisePlugin::OptixDenoiseContext*>( ptr );
    delete ctx;
}

struct OptixDenoiseEventData {
    intptr_t denoiseContext;
    intptr_t albedo;
    intptr_t normal;
    intptr_t flow;
    intptr_t color;
    intptr_t output;
    uint32_t readback;
    intptr_t readbackTexture;
};

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API OnRenderEvent( int eventId, void* data )
{
    OptixDenoiseEventData* event = reinterpret_cast<OptixDenoiseEventData*>( data );

    UnityDenoisePlugin::OptixDenoiseImageData imageData;
    imageData.albedo = reinterpret_cast<ID3D12Resource*>( event->albedo );
    imageData.normal = reinterpret_cast<ID3D12Resource*>( event->normal );
    imageData.flow = reinterpret_cast<ID3D12Resource*>( event->flow );
    imageData.color = reinterpret_cast<ID3D12Resource*>( event->color );
    imageData.output = reinterpret_cast<ID3D12Resource*>( event->output );
    imageData.readback = static_cast<UnityDenoisePlugin::OptixDenoiseReadback>( event->readback );
    imageData.readbackTexture = reinterpret_cast<ID3D12Resource*>( event->readbackTexture );

    auto denoseContext = reinterpret_cast<UnityDenoisePlugin::OptixDenoiseContext*>( event->denoiseContext );
    denoseContext->Denoise( imageData );
}

// Freely defined function to pass a callback to plugin-specific scripts
extern "C" UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc()
{
    return OnRenderEvent;
}

