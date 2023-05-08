using System;
using System.Runtime.InteropServices;


namespace UnityDenoiserPlugin
{
    public enum DenoiserType
    {
        Optix = 0,
        Oidn = 1,
    }

    public enum Readback
    {
        None = 0,
        Albedo = 1,
        Normal = 2,
        Flow = 3,
        Color = 4,
        PreviousOutput = 5,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DenoiseConfig
    {
        public uint imageWidth;
        public uint imageHeight;
        public uint guideAlbedo;
        public uint guideNormal;
        public uint temporalMode;

        public bool Equals(DenoiseConfig cfg)
        {
            return imageWidth == cfg.imageWidth &&
                   imageHeight == cfg.imageHeight &&
                   guideAlbedo == cfg.guideAlbedo &&
                   guideNormal == cfg.guideNormal &&
                   temporalMode == cfg.temporalMode;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DenoiseEventData
    {
        public IntPtr denoiseContext;
        public IntPtr albedo;
        public IntPtr normal;
        public IntPtr flow;
        public IntPtr color;
        public IntPtr output;
        public Readback readback;
        public IntPtr readbackTexture;
    }

    public static class Interface
    {
        [DllImport("UnityDenoiserPlugin")]
        public static extern IntPtr CreateDenoiseContext(DenoiserType type, ref DenoiseConfig cfg);

        [DllImport("UnityDenoiserPlugin")]
        public static extern void DestroyDenoiseContext(DenoiserType type, IntPtr ptr);

        [DllImport("UnityDenoiserPlugin")]
        public static extern IntPtr GetRenderEventFunc();

        [DllImport("UnityDenoiserPlugin")]
        public static extern void OIDNSetPluginsAndWeightsFolder(string pluginsFolder, string baseWeightFolder);
    }
}

