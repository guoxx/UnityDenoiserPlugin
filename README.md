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

## Best Practices

NVIDIA OptiX Denoiser has built-in support for very dark and bright images. However, OIDN relies on the application side to provide a well-exposed image. If you use OIDN, make sure to pre-expose the input color image before denoising, and then apply the inverse pre-exposure to obtain the final result.

## Future Work

The current implementation is not optimal for a production renderer, as it requires copying data between D3D and CUDA multiple times. This limitation stems from the fact that Unity-created resources do not have shared flags when creating D3D resources. Future work could focus on improving the data transfer process to optimize performance and reduce copying overhead.

## Special Thanks

We want to thank [@maxlianli](https://twitter.com/maxliani) for the great blog posts that inspired this project. We really recommend checking out their easy-to-understand deep learning articles:

1. [DNND-1: Exploring Deep Neural Networks](https://maxliani.wordpress.com/2023/03/17/dnnd-1-a-deep-neural-network-dive/)
2. [DNND-2: Learning about Tensors and Convolution](https://maxliani.wordpress.com/2023/03/24/dnnd-2-tensors-and-convolution/)
3. [DNND-3: Understanding the U-Net Architecture](https://maxliani.wordpress.com/2023/04/07/dnnd-3-the-u-net-architecture/)
