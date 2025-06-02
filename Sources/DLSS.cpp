#include "DLSS.h"
#include "RHI.h" 
#include "Utils.h"

// NVIDIA NGX SDK headers
#include "nvsdk_ngx.h" 
#include "nvsdk_ngx_defs.h" 
#include "nvsdk_ngx_defs_dlssd.h" 

#include <sstream>
#include <iomanip>
#include <unordered_map>


extern "C" {

// Helper function to convert NVSDK_NGX_Feature to string
const char* GetFeatureString(NVSDK_NGX_Feature feature)
{
    switch (feature)
    {
        case NVSDK_NGX_Feature_SuperSampling:
            return "DLSS";
        case NVSDK_NGX_Feature_InPainting:
            return "InPainting";
        case NVSDK_NGX_Feature_ImageSuperResolution:
            return "ImageSuperResolution";
        case NVSDK_NGX_Feature_SlowMotion:
            return "SlowMotion";
        case NVSDK_NGX_Feature_VideoSuperResolution:
            return "VideoSuperResolution";
        case NVSDK_NGX_Feature_ImageSignalProcessing:
            return "ImageSignalProcessing";
        case NVSDK_NGX_Feature_DeepResolve:
            return "DeepResolve";
        case NVSDK_NGX_Feature_FrameGeneration:
            return "FrameGeneration";
        case NVSDK_NGX_Feature_DeepDVC:
            return "DeepDVC";
        case NVSDK_NGX_Feature_RayReconstruction:
            return "RayReconstruction";
        case NVSDK_NGX_Feature_Reserved_SDK:
            return "SDK";
        case NVSDK_NGX_Feature_Reserved_Core:
            return "Core";
        case NVSDK_NGX_Feature_Reserved_Unknown:
            return "Unknown";
        default:
            return "Reserved";
    }
}

// Helper function to log DLSS errors
void LogDlssResult(NVSDK_NGX_Result result, const char* functionName)
{
    if (!NVSDK_NGX_SUCCEED(result))
    {
        std::ostringstream oss;
        oss << "[DLSS] " << functionName << " failed with error code: " << result;
        
        // Add more specific information based on error code
        switch (result)
        {
        case NVSDK_NGX_Result_FAIL_FeatureNotSupported:
            oss << " - Feature not supported on current hardware";
            break;
        case NVSDK_NGX_Result_FAIL_PlatformError:
            oss << " - Platform error, check D3D12 debug layer for more info";
            break;
        case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists:
            oss << " - Feature with given parameters already exists";
            break;
        case NVSDK_NGX_Result_FAIL_FeatureNotFound:
            oss << " - Feature with provided handle does not exist";
            break;
        case NVSDK_NGX_Result_FAIL_InvalidParameter:
            oss << " - Invalid parameter was provided";
            break;
        case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall:
            oss << " - Provided buffer is too small";
            break;
        case NVSDK_NGX_Result_FAIL_NotInitialized:
            oss << " - SDK was not initialized properly";
            break;
        case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat:
            oss << " - Unsupported format used for input/output buffers";
            break;
        case NVSDK_NGX_Result_FAIL_RWFlagMissing:
            oss << " - Feature input/output needs RW access (UAV)";
            break;
        case NVSDK_NGX_Result_FAIL_MissingInput:
            oss << " - Feature was created with specific input but none is provided at evaluation";
            break;
        case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:
            oss << " - Feature is not available on the system";
            break;
        case NVSDK_NGX_Result_FAIL_OutOfDate:
            oss << " - NGX system libraries are old and need an update";
            break;
        case NVSDK_NGX_Result_FAIL_OutOfGPUMemory:
            oss << " - Feature requires more GPU memory than is available";
            break;
        case NVSDK_NGX_Result_FAIL_UnsupportedFormat:
            oss << " - Format used in input buffer(s) is not supported by feature";
            break;
        case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath:
            oss << " - Path provided in InApplicationDataPath cannot be written to";
            break;
        case NVSDK_NGX_Result_FAIL_UnsupportedParameter:
            oss << " - Unsupported parameter was provided";
            break;
        case NVSDK_NGX_Result_FAIL_Denied:
            oss << " - The feature or application was denied (contact NVIDIA for details)";
            break;
        case NVSDK_NGX_Result_FAIL_NotImplemented:
            oss << " - The feature or functionality is not implemented";
            break;
        default:
            oss << " - Unknown error";
            break;
        }
        
        UnityDenoiserPlugin::LogError(oss.str().c_str());
    }
}

// DLSS SDK Log Callback
void NVSDK_CONV DLSSLogCallback(const char* message, NVSDK_NGX_Logging_Level loggingLevel, NVSDK_NGX_Feature sourceComponent)
{
    // Format the message with component information
    std::ostringstream oss;
    oss << "[DLSS][" << GetFeatureString(sourceComponent) << "]: " << message;
    
    // Route to the appropriate Unity logging function based on severity
    switch (loggingLevel)
    {
    case NVSDK_NGX_LOGGING_LEVEL_VERBOSE:
    case NVSDK_NGX_LOGGING_LEVEL_ON:
        UnityDenoiserPlugin::LogMessage(oss.str().c_str());
        break;
    case NVSDK_NGX_LOGGING_LEVEL_OFF: // This shouldn't happen, but handle it anyway
        // No logging
        break;
    default:
        UnityDenoiserPlugin::LogWarning(oss.str().c_str());
        break;
    }
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Init_with_ProjectID_D3D12(
    const DLSSInitParams* params
)
{
    if (!params)
    {
        UnityDenoiserPlugin::LogError("DLSS_Init_with_ProjectID_D3D12: params is null");
        return static_cast<int>(NVSDK_NGX_Result_FAIL_InvalidParameter);
    }

    // Cast the generic void* to ID3D12Device*
    ID3D12Device* pd3d12Device = UnityDenoiserPlugin::RHI::GetDevice();

    // Create feature common info for logging
    NVSDK_NGX_FeatureCommonInfo featureInfo = {};
    
    // Setup logging info
    featureInfo.LoggingInfo.LoggingCallback = DLSSLogCallback;
    featureInfo.LoggingInfo.MinimumLoggingLevel = params->loggingLevel;
    featureInfo.LoggingInfo.DisableOtherLoggingSinks = true;

    // Call the NGX SDK function
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init_with_ProjectID(
        params->projectId,
        params->engineType,
        params->engineVersion,
        params->applicationDataPath,
        pd3d12Device,
        &featureInfo, // Using our featureInfo with logging callback
        NVSDK_NGX_Version_API 
    );

    return static_cast<int>(result);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Shutdown_D3D12()
{
    ID3D12Device* pd3d12Device = UnityDenoiserPlugin::RHI::GetDevice();
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Shutdown1(pd3d12Device);
    return static_cast<int>(result);
}

static uint32_t g_dlssFeatureHandleCounter = 0;
static std::unordered_map<uint32_t, NVSDK_NGX_Handle*> g_dlssFeatureHandles;

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_AllocateFeatureHandle()
{
    // TODO: find better allocation strategy
    int handle = g_dlssFeatureHandleCounter % 1024;
    if ( g_dlssFeatureHandles.find(handle) != g_dlssFeatureHandles.end() )
    {
        UnityDenoiserPlugin::LogError("DLSS_AllocateFeatureHandle: handle already exists");
        return -1;
    }

    g_dlssFeatureHandles[handle] = nullptr;
    g_dlssFeatureHandleCounter += 1;
    return handle;
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_FreeFeatureHandle(int handle)
{
    if ( g_dlssFeatureHandles.find(handle) == g_dlssFeatureHandles.end() )
    {
        UnityDenoiserPlugin::LogError("DLSS_FreeFeatureHandle: handle does not exist");
        return -1;
    }

    g_dlssFeatureHandles.erase(handle);
    return 0;
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API OnDLSSRenderEvent( int eventId, void* data )
{
    if ( eventId == 0 )
    {
        DLSSCreateFeatureParams *params = reinterpret_cast<DLSSCreateFeatureParams*>( data );

        UnityDenoiserPlugin::RHI::CommandListChunk cmdlist = UnityDenoiserPlugin::RHI::GetCommandList();

        NVSDK_NGX_Handle *ngxHandle = nullptr;
        NVSDK_NGX_Result result = NVSDK_NGX_D3D12_CreateFeature(
            cmdlist.commandList,
            params->feature,
            params->parameters,
            &ngxHandle);
        LogDlssResult(result, "NVSDK_NGX_D3D12_CreateFeature");

        if (NVSDK_NGX_SUCCEED(result))
        {
            UnityDenoiserPlugin::RHI::ExecuteCommandList(cmdlist, 0, nullptr);
            g_dlssFeatureHandles[params->handle] = ngxHandle;
        }
        else
        {
            UnityDenoiserPlugin::RHI::RecycleCommandList(cmdlist);
        }
    }
    else if ( eventId == 1 )
    {
        DLSSEvaluateFeatureParams *params = reinterpret_cast<DLSSEvaluateFeatureParams*>( data );
        NVSDK_NGX_Handle *ngxHandle = g_dlssFeatureHandles[params->handle];

        if ( ngxHandle != nullptr )
        {
            UnityDenoiserPlugin::RHI::CommandListChunk cmdlist = UnityDenoiserPlugin::RHI::GetCommandList();

            NVSDK_NGX_Result result = NVSDK_NGX_D3D12_EvaluateFeature( cmdlist.commandList, ngxHandle, params->parameters );
            LogDlssResult( result, "NVSDK_NGX_D3D12_EvaluateFeature" );

            if (NVSDK_NGX_SUCCEED(result))
            {
                UnityDenoiserPlugin::RHI::ExecuteCommandList( cmdlist, 0, nullptr );
            }
            else
            {
                UnityDenoiserPlugin::RHI::RecycleCommandList(cmdlist);
            }
        }
        else
        {
            UnityDenoiserPlugin::LogError("DLSS_EvaluateFeature: ngxHandle is null");
        }
    }
    else if ( eventId == 2 )
    {
        DLSSDestroyFeatureParams *params = reinterpret_cast<DLSSDestroyFeatureParams*>( data );
        NVSDK_NGX_Handle *ngxHandle = g_dlssFeatureHandles[params->handle];

        if ( ngxHandle != nullptr )
        {
            NVSDK_NGX_Result result = NVSDK_NGX_D3D12_ReleaseFeature(ngxHandle);
            LogDlssResult(result, "NVSDK_NGX_D3D12_ReleaseFeature");
        }
        else
        {
            UnityDenoiserPlugin::LogError("DLSS_DestroyFeature: ngxHandle is null");
        }

        DLSS_FreeFeatureHandle(params->handle);
    }
    else
    {
        UnityDenoiserPlugin::LogError("Unknown DLSS event id");
    }
}

UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_UnityRenderEventFunc()
{
    return OnDLSSRenderEvent;
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_AllocateParameters_D3D12(
    NVSDK_NGX_Parameter** ppOutParameters
)
{
    return NVSDK_NGX_D3D12_AllocateParameters(ppOutParameters);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_GetCapabilityParameters_D3D12(
    NVSDK_NGX_Parameter** ppOutParameters
)
{
    return NVSDK_NGX_D3D12_GetCapabilityParameters(ppOutParameters);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_DestroyParameters_D3D12(
    NVSDK_NGX_Parameter* pInParameters
)
{
    return NVSDK_NGX_D3D12_DestroyParameters(pInParameters);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetULL(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    unsigned long long value)
{
    NVSDK_NGX_Parameter_SetULL(pParameters, paramName, value);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetF(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    float value)
{
    NVSDK_NGX_Parameter_SetF(pParameters, paramName, value);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetD(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    double value)
{
    NVSDK_NGX_Parameter_SetD(pParameters, paramName, value);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetUI(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    unsigned int value)
{
    NVSDK_NGX_Parameter_SetUI(pParameters, paramName, value);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetI(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    int value)
{
    NVSDK_NGX_Parameter_SetI(pParameters, paramName, value);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetD3d11Resource(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    struct ID3D11Resource* value)
{
    NVSDK_NGX_Parameter_SetD3d11Resource(pParameters, paramName, value);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetD3d12Resource(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    struct ID3D12Resource* value)
{
    NVSDK_NGX_Parameter_SetD3d12Resource(pParameters, paramName, value);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetVoidPointer(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    void* value)
{
    NVSDK_NGX_Parameter_SetVoidPointer(pParameters, paramName, value);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetULL(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    unsigned long long* pValue)
{
    return NVSDK_NGX_Parameter_GetULL(pParameters, paramName, pValue);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetF(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    float* pValue)
{
    return NVSDK_NGX_Parameter_GetF(pParameters, paramName, pValue);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetD(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    double* pValue)
{
    return NVSDK_NGX_Parameter_GetD(pParameters, paramName, pValue);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetUI(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    unsigned int* pValue)
{
    return NVSDK_NGX_Parameter_GetUI(pParameters, paramName, pValue);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetI(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    int* pValue)
{
    return NVSDK_NGX_Parameter_GetI(pParameters, paramName, pValue);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetD3d11Resource(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    struct ID3D11Resource** ppValue)
{
    return NVSDK_NGX_Parameter_GetD3d11Resource(pParameters, paramName, ppValue);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetD3d12Resource(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    struct ID3D12Resource** ppValue)
{
    return NVSDK_NGX_Parameter_GetD3d12Resource(pParameters, paramName, ppValue);
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetVoidPointer(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    void** ppValue)
{
    return NVSDK_NGX_Parameter_GetVoidPointer(pParameters, paramName, ppValue);
}

} // extern "C"
