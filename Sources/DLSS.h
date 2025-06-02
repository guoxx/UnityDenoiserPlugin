#pragma once

#include "IUnityGraphics.h"
#include "nvsdk_ngx_params.h"
#include "nvsdk_ngx_defs.h"


#ifdef __cplusplus
extern "C" {
#endif

struct DLSSInitParams
{
    const char* projectId;
    NVSDK_NGX_EngineType engineType;
    const char* engineVersion;
    const wchar_t* applicationDataPath;
    NVSDK_NGX_Logging_Level loggingLevel;
};

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Init_with_ProjectID_D3D12(
    const DLSSInitParams* params
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Shutdown_D3D12();

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_AllocateParameters_D3D12(
    NVSDK_NGX_Parameter** ppOutParameters
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_GetCapabilityParameters_D3D12(
    NVSDK_NGX_Parameter** ppOutParameters
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_DestroyParameters_D3D12(
    NVSDK_NGX_Parameter* pInParameters
);

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetULL(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    unsigned long long value
);

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetF(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    float value
);

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetD(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    double value
);

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetUI(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    unsigned int value
);

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetI(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    int value
);

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetD3d11Resource(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    struct ID3D11Resource* value
);

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetD3d12Resource(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    struct ID3D12Resource* value
);

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_SetVoidPointer(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    void* value
);

// Getters
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetULL(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    unsigned long long* pValue
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetF(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    float* pValue
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetD(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    double* pValue
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetUI(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    unsigned int* pValue
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetI(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    int* pValue
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetD3d11Resource(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    struct ID3D11Resource** ppValue
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetD3d12Resource(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    struct ID3D12Resource** ppValue
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_Parameter_GetVoidPointer(
    NVSDK_NGX_Parameter* pParameters,
    const char* paramName,
    void** ppValue
);

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_AllocateFeatureHandle();

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_FreeFeatureHandle(int handle);

struct DLSSCreateFeatureParams
{
    int handle;
    NVSDK_NGX_Feature feature;
    NVSDK_NGX_Parameter *parameters;
};

struct DLSSEvaluateFeatureParams
{
    int handle;
    NVSDK_NGX_Parameter *parameters;
};

struct DLSSDestroyFeatureParams
{
    int handle;
};

UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API DLSS_UnityRenderEventFunc();

#ifdef __cplusplus
} // extern "C"
#endif
