using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityDenoiserPlugin;
using static UnityDenoiserPlugin.DlssSdk;

namespace UnityDenoiserPlugin
{
    public sealed class DLSSSuperSampling : IRenderPass
    {
        private int m_dlssHandle = DLSS_INVALID_FEATURE_HANDLE;
        private IntPtr m_dlssParameters = IntPtr.Zero;
        private bool m_dlssInitialized = false;

        private NVSDK_NGX_DLSS_Create_Params m_dlssCreateParams;
        private bool m_createParamsChanged = false;

        public DLSSSuperSampling(NVSDK_NGX_DLSS_Feature_Flags featureCreateFlags,
                                 NVSDK_NGX_PerfQuality_Value perfQualityValue)
        {
            DLSS_Init();

            m_dlssCreateParams = new NVSDK_NGX_DLSS_Create_Params();
            m_dlssCreateParams.Feature = new NVSDK_NGX_Feature_Create_Params();
            m_dlssCreateParams.Feature.InPerfQualityValue = perfQualityValue;
            m_dlssCreateParams.InFeatureCreateFlags = (int)featureCreateFlags;
            m_dlssCreateParams.InEnableOutputSubrects = false;
            // Initialize with default values, will be updated when first used
            m_dlssCreateParams.Feature.InWidth = 0;
            m_dlssCreateParams.Feature.InHeight = 0;
            m_dlssCreateParams.Feature.InTargetWidth = 0;
            m_dlssCreateParams.Feature.InTargetHeight = 0;
        }

        protected override void Dispose(bool disposing)
        {
            if (m_disposed)
                return;

            if (disposing)
            {
                using var commands = new ScopedCommandBuffer("DLSS Super Sampling Cleanup");
                DisposeDLSSResources(commands);
            }

            DLSS_Shutdown();

            base.Dispose(disposing);
        }

        private void DisposeDLSSResources(CommandBuffer commands)
        {
            if (m_dlssInitialized)
            {
                if (m_dlssHandle != DLSS_INVALID_FEATURE_HANDLE)
                {
                    DLSS_DestroyFeature(commands, m_dlssHandle);
                    m_dlssHandle = DLSS_INVALID_FEATURE_HANDLE;
                }

                if (m_dlssParameters != IntPtr.Zero)
                {
                    DLSS_DestroyParameters_D3D12(m_dlssParameters);
                    m_dlssParameters = IntPtr.Zero;
                }

                m_dlssInitialized = false;
            }
        }

        private bool InitializeDLSSSuperSampling(CommandBuffer cmd)
        {
            if (m_dlssInitialized)
                return true;

            // Check if DLSS is available
            if (!DLSS_IsSuperSamplingAvailable())
            {
                Debug.LogError("DLSS Super Sampling is not available");
                return false;
            }

            // Allocate DLSS parameters
            var result = DLSS_AllocateParameters_D3D12(out m_dlssParameters);
            if (!NVSDK_NGX_SUCCEED(result))
            {
                Debug.LogError($"Failed to allocate DLSS parameters: {result}");
                return false;
            }

            // Create DLSS Super Sampling Feature
            NGX_D3D12_CREATE_DLSS_EXT(cmd, 1, 1, out m_dlssHandle, m_dlssParameters, ref m_dlssCreateParams);

            if (m_dlssHandle == DLSS_INVALID_FEATURE_HANDLE)
            {
                Debug.LogError("Failed to create DLSS Super Sampling Feature");
                DLSS_DestroyParameters_D3D12(m_dlssParameters);
                m_dlssParameters = IntPtr.Zero;
                return false;
            }

            m_dlssInitialized = true;
            return true;
        }

        private void CheckAndResetDLSS(CommandBuffer commands, NVSDK_NGX_D3D12_DLSS_Eval_Params dlssEvalParams)
        {
            if (dlssEvalParams.Feature.pInColor.width != m_dlssCreateParams.Feature.InWidth ||
                dlssEvalParams.Feature.pInColor.height != m_dlssCreateParams.Feature.InHeight ||
                dlssEvalParams.Feature.pInOutput.width != m_dlssCreateParams.Feature.InTargetWidth ||
                dlssEvalParams.Feature.pInOutput.height != m_dlssCreateParams.Feature.InTargetHeight)
            {
                m_createParamsChanged = true;

                m_dlssCreateParams.Feature.InWidth = (uint)dlssEvalParams.Feature.pInColor.width;
                m_dlssCreateParams.Feature.InHeight = (uint)dlssEvalParams.Feature.pInColor.height;
                m_dlssCreateParams.Feature.InTargetWidth = (uint)dlssEvalParams.Feature.pInOutput.width;
                m_dlssCreateParams.Feature.InTargetHeight = (uint)dlssEvalParams.Feature.pInOutput.height;
            }

            if (m_createParamsChanged)
            {
                DisposeDLSSResources(commands);
            }
            m_createParamsChanged = false;
        }

        public bool IsSupported()
        {
            return DLSS_IsSuperSamplingAvailable();
        }

        public void SetFeatureCreateFlags(NVSDK_NGX_DLSS_Feature_Flags featureCreateFlags)
        {
            if (m_dlssCreateParams.InFeatureCreateFlags == (int)featureCreateFlags)
                return;

            m_dlssCreateParams.InFeatureCreateFlags = (int)featureCreateFlags;
            m_createParamsChanged = true;
        }

        public void SetPerformanceQuality(NVSDK_NGX_PerfQuality_Value perfQualityValue)
        {
            if (m_dlssCreateParams.Feature.InPerfQualityValue == perfQualityValue)
                return;

            m_dlssCreateParams.Feature.InPerfQualityValue = perfQualityValue;
            m_createParamsChanged = true;
        }

        public bool Render(CommandBuffer commands, NVSDK_NGX_D3D12_DLSS_Eval_Params dlssEvalParams)
        {
            CheckAndResetDLSS(commands, dlssEvalParams);

            // Initialize DLSS Super Sampling
            if (InitializeDLSSSuperSampling(commands))
            {
                // Execute DLSS Super Sampling
                NGX_D3D12_EVALUATE_DLSS_EXT(commands, m_dlssHandle, m_dlssParameters, ref dlssEvalParams);
                return true;
            }

            return false;
        }
    }
} 