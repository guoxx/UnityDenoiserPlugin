using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityDenoiserPlugin;
using static UnityDenoiserPlugin.DlssSdk;

namespace UnityDenoiserPlugin
{
    public sealed class DLSSRayReconstruction : IRenderPass
    {
        private int m_dlssHandle = DLSS_INVALID_FEATURE_HANDLE;
        private IntPtr m_dlssParameters = IntPtr.Zero;
        private bool m_dlssInitialized = false;

        private NVSDK_NGX_DLSSD_Create_Params m_dlssCreateParams;
        private bool m_createParamsChanged = false;

        public DLSSRayReconstruction(NVSDK_NGX_DLSS_Feature_Flags featureCreateFlags,
                                     NVSDK_NGX_PerfQuality_Value perfQualityValue,
                                     NVSDK_NGX_DLSS_Roughness_Mode roughnessMode,
                                     NVSDK_NGX_DLSS_Depth_Type depthType)
        {
            DLSS_Init();

            m_dlssCreateParams = new NVSDK_NGX_DLSSD_Create_Params();
            m_dlssCreateParams.InPerfQualityValue = perfQualityValue;
            m_dlssCreateParams.InFeatureCreateFlags = (int)featureCreateFlags;
            m_dlssCreateParams.InRoughnessMode = roughnessMode;
            m_dlssCreateParams.InUseHWDepth = depthType;
            m_dlssCreateParams.InEnableOutputSubrects = 0;
            // Initialize with default values, will be updated when first used
            m_dlssCreateParams.InWidth = 0;
            m_dlssCreateParams.InHeight = 0;
            m_dlssCreateParams.InTargetWidth = 0;
            m_dlssCreateParams.InTargetHeight = 0;
        }

        protected override void Dispose(bool disposing)
        {
            if (m_disposed)
                return;

            if (disposing)
            {
                using var commands = new ScopedCommandBuffer("DLSS Cleanup");
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

        private bool InitializeDLSSRayReconstruction(CommandBuffer cmd)
        {
            if (m_dlssInitialized)
                return true;

            // Check if DLSS Ray Reconstruction is available
            if (!DLSS_IsRayReconstructionAvailable())
            {
                Debug.LogError("DLSS Ray Reconstruction is not available");
                return false;
            }

            // Allocate DLSS parameters
            var result = DLSS_AllocateParameters_D3D12(out m_dlssParameters);
            if (!NVSDK_NGX_SUCCEED(result))
            {
                Debug.LogError($"Failed to allocate DLSS parameters: {result}");
                return false;
            }

            // Create DLSS Ray Reconstruction Feature
            NGX_D3D12_CREATE_DLSSD_EXT(cmd, 1, 1, out m_dlssHandle, m_dlssParameters, m_dlssCreateParams);

            if (m_dlssHandle == DLSS_INVALID_FEATURE_HANDLE)
            {
                Debug.LogError("Failed to create DLSS Ray Reconstruction Feature");
                DLSS_DestroyParameters_D3D12(m_dlssParameters);
                m_dlssParameters = IntPtr.Zero;
                return false;
            }

            m_dlssInitialized = true;
            return true;
        }

        private void CheckAndResetDLSS(CommandBuffer commands, NVSDK_NGX_D3D12_DLSSD_Eval_Params dlssEvalParams)
        {
            if (dlssEvalParams.pInColor.width != m_dlssCreateParams.InWidth ||
                dlssEvalParams.pInColor.height != m_dlssCreateParams.InHeight ||
                dlssEvalParams.pInOutput.width != m_dlssCreateParams.InTargetWidth ||
                dlssEvalParams.pInOutput.height != m_dlssCreateParams.InTargetHeight)
            {
                m_createParamsChanged = true;

                m_dlssCreateParams.InWidth = (uint)dlssEvalParams.pInColor.width;
                m_dlssCreateParams.InHeight = (uint)dlssEvalParams.pInColor.height;
                m_dlssCreateParams.InTargetWidth = (uint)dlssEvalParams.pInOutput.width;
                m_dlssCreateParams.InTargetHeight = (uint)dlssEvalParams.pInOutput.height;
            }

            if (m_createParamsChanged)
            {
                DisposeDLSSResources(commands);
            }
            m_createParamsChanged = false;
        }

        public bool IsSupported()
        {
            return DLSS_IsRayReconstructionAvailable();
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
            if (m_dlssCreateParams.InPerfQualityValue == perfQualityValue)
                return;

            m_dlssCreateParams.InPerfQualityValue = perfQualityValue;
            m_createParamsChanged = true;
        }

        public void SetRoughnessMode(NVSDK_NGX_DLSS_Roughness_Mode roughnessMode)
        {
            if (m_dlssCreateParams.InRoughnessMode == roughnessMode)
                return;

            m_dlssCreateParams.InRoughnessMode = roughnessMode;
            m_createParamsChanged = true;
        }

        public void SetDepthType(NVSDK_NGX_DLSS_Depth_Type depthType)
        {
            if (m_dlssCreateParams.InUseHWDepth == depthType)
                return;

            m_dlssCreateParams.InUseHWDepth = depthType;
            m_createParamsChanged = true;
        }

        public bool Render(CommandBuffer commands, NVSDK_NGX_D3D12_DLSSD_Eval_Params dlssEvalParams)
        {
            CheckAndResetDLSS(commands, dlssEvalParams);

            // Initialize DLSS Ray Reconstruction
            if (InitializeDLSSRayReconstruction(commands))
            {
                // Execute DLSS Ray Reconstruction
                NGX_D3D12_EVALUATE_DLSSD_EXT(commands, m_dlssHandle, m_dlssParameters, dlssEvalParams);
                return true;
            }

            return false;
        }
    }
}
