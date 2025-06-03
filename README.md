# UnityDenoiserPlugin

UnityDenoiserPlugin is an open-source implementation of a real-time denoiser for Unity. It comes with two backends: NVIDIA OptiX Denoiser and Intel's Open Image Denoise (OIDN).

The plugin is built upon D3D12 and uses CUDA interpolation to efficiently synchronize GPU execution between D3D command lists and CUDA streams. This ensures smooth performance and concurrency in your projects.

While UnityDenoiserPlugin has been specifically designed for Unity, it can be easily adapted to any D3D12 project with just a few minor modifications, making it a flexible and convenient solution for a wide range of D3D12 based applications.

https://user-images.githubusercontent.com/1138365/236827645-213db9e0-c20c-4eb1-aa6b-d2aa4183d123.mp4


## Differences between UnityDenoiserPlugin and com.unity.rendering.denoising (UDN)

UnityDenoiserPlugin is designed for real-time performance, taking only a few milliseconds to denoise an image. It doesn't require data synchronization between CPU and GPU, as everything is done on the GPU using D3D12 and CUDA interpolation.

## How to Compile

1. Install [NVIDIA OptiX](https://developer.nvidia.com/optix) and set the `OptiX_INSTALL_DIR` environment variable to the correct directory.
2. Compile the source code with CMake.

## DLSS Integration

This plugin supports NVIDIA DLSS, including DLSS Super Sampling (SR) and DLSS Ray Reconstruction (RR).

### General Initialization and Cleanup

Manage DLSS SDK lifecycle within your application or rendering module:
*   Initialization: `DLSS_Init()`
*   Cleanup: `DLSS_Shutdown()`

These functions use a reference count. Ensure `DLSS_Init` calls are matched by `DLSS_Shutdown` calls.

### Integrating DLSS Features (SR & RR)

The core process for integrating DLSS features (both Super Sampling and Ray Reconstruction) involves these main steps. It's recommended to integrate and test SR before RR.

1.  Check Availability:
    *   For Super Sampling (SR): `DLSS_IsSuperSamplingAvailable()`
    *   For Ray Reconstruction (RR): `DLSS_IsRayReconstructionAvailable()`

2.  Parameter Allocation & Feature Creation: This is typically done when resolution or quality settings change.
    *   Allocate DLSS parameters using `DLSS_AllocateParameters_D3D12()`. This can be shared between SR and RR.
    *   Create the specific DLSS feature:
        *   **SR:** Use `NGX_D3D12_CREATE_DLSS_EXT()`. Key settings in `NVSDK_NGX_DLSS_Create_Params` include input/output resolutions, performance quality mode, and feature flags.
        *   **RR:** Use `NGX_D3D12_CREATE_DLSSD_EXT()`. Key settings in `NVSDK_NGX_DLSSD_Create_Params` include input/output resolutions (often the same for RR), quality mode, feature flags, G-Buffer roughness mode, and hardware depth usage.

3.  Feature Evaluation (Per Frame): Execute the DLSS processing in each frame.
    *   **SR:** Call `NGX_D3D12_EVALUATE_DLSS_EXT()`. Inputs via `NVSDK_NGX_D3D12_DLSS_Eval_Params` include the low-resolution color, depth, motion vectors, jitter, and sharpness. Outputs the upscaled image.
    *   **RR:** Call `NGX_D3D12_EVALUATE_DLSSD_EXT()`. In addition to inputs similar to SR, `NVSDK_NGX_D3D12_DLSSD_Eval_Params` requires G-Buffer data (diffuse albedo, specular albedo, normals, roughness) and view/projection matrices.

4.  Resource Management: Proper management of DLSS resources is crucial. Refer to the "Resource Management and Resolution Changes" section below for details on releasing and re-creating features upon resolution changes.

### Resource Management and Resolution Changes

*   Release Resources: When a feature is no longer needed or parameters change, release old resources:
    *   `DLSS_DestroyFeature(CommandBuffer cmd, int dlssHandle)`
    *   `DLSS_DestroyParameters_D3D12(IntPtr dlssParameters)`
*   Handle Resolution Changes: On render/output size changes, re-allocate parameters and recreate features with new dimensions (release old ones first).

### DLSS Best Practices & Key Considerations

*   Prioritize SR then RR: Strongly recommended to integrate and validate DLSS Super Sampling with correct basic inputs (motion vectors, depth, jitter) before integrating DLSS Ray Reconstruction.
*   Consult NVIDIA Official Documentation: For an in-depth understanding of parameters, flags, and best practices, always refer to the official DLSS documentation in the NVIDIA DLSS SDK. This guide is for a quick start.
*   DLSS Logging: All messages from the DLSS SDK (errors and logs) might appear as standard Unity Log Info messages. Check the Unity console carefully when debugging DLSS issues.
*   DLSS SDK Debug Tools: The NVIDIA DLSS SDK often includes debug tools (e.g., registry modification tools) that can be helpful. Refer to the DLSS SDK documentation.

## Best Practices

NVIDIA OptiX Denoiser has built-in support for very dark and bright images. However, OIDN relies on the application side to provide a well-exposed image. If you use OIDN, make sure to pre-expose the input color image before denoising, and then apply the inverse pre-exposure to obtain the final result.

## Future Work

The current implementation is not optimal for a production renderer, as it requires copying data between D3D and CUDA multiple times. This limitation stems from the fact that Unity-created resources do not have shared flags when creating D3D resources. Future work could focus on improving the data transfer process to optimize performance and reduce copying overhead.

## Special Thanks

We want to thank [@maxlianli](https://twitter.com/maxliani) for the great blog posts that inspired this project. We really recommend checking out their easy-to-understand deep learning articles:

1. [DNND-1: Exploring Deep Neural Networks](https://maxliani.wordpress.com/2023/03/17/dnnd-1-a-deep-neural-network-dive/)
2. [DNND-2: Learning about Tensors and Convolution](https://maxliani.wordpress.com/2023/03/24/dnnd-2-tensors-and-convolution/)
3. [DNND-3: Understanding the U-Net Architecture](https://maxliani.wordpress.com/2023/04/07/dnnd-3-the-u-net-architecture/)
