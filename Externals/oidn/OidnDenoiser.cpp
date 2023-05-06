// This file was automatically generated, but should be customized/replaced to meet your needs.
#include "OidnDenoiser.h"
#include "export/BuildNetwork_OIDN.h"
#include <CLI11/CLI11.hpp>
#include <ClassRegistry.h>
#include <CpuImage.h>
#include <DynamicPluginLoader.h>
#include <FilesystemWeightsLoader.h>
#include <LayerListIterators.h>
#include <SimpleLogger.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <npy/npy.hpp>
#include <nvneural/CudaTypes.h>
#include <nvneural/NetworkTypes.h>
#include <nvneural/layers/IStandardInputLayer.h>
#include <nvneural/layers/ICudaInputLayer.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "SimpleLogger.h"
#include "CopyImage.cuh"

using namespace nvneural;

namespace oidn
{
    class OidnDenoiserImpl
    {
    public:
        RefPtr<INetwork> pNetwork = {};
        RefPtr<IClassRegistry> pRegistry = {};
        RefPtr<oidn::SimpleLogger> pLogger = {};
        RefPtr<INetworkBackend> pCudaBackend = {};
        RefPtr<IWeightsLoader> pWeightsLoader = {};
        RefPtr<IPluginLoader> pPluginLoader = {};

        CUstream stream_ = nullptr;

        size_t inputDataSize_ = 0;
        void* pInputData_ = nullptr;

        // Tensor format defaults are based on how ConverenceNG exported the network.
        bool fp16 = false;
        bool nhwc = false;
        int cudaDeviceIndex = 0;
        int passCount = 1;
#if defined(_DEBUG) || defined(DEBUG)
        int verbosity = 2;
#else
        int verbosity = 0;
#endif
        std::string layerName_input = "input";
        std::string layerName_output = "output";

        OidnDenoiserImpl(bool guideAlbedo, bool guideNormal, const std::string& baseWeightsPath, const std::string& pluginsFolder, LogCallback logCallback)
        {
            NeuralResult status;

            pLogger = make_refobject<oidn::SimpleLogger>(verbosity).as<oidn::SimpleLogger>();
            pLogger->setLogCallback(logCallback);
            SetDefaultLogger(pLogger.get());

            const TensorDataType tensorType = fp16 ? TensorDataType::Half : TensorDataType::Float;
            const TensorDataLayout tensorLayout = nhwc ? TensorDataLayout::Nhwc : TensorDataLayout::Nchw;
            const TensorFormat tensorFormat = {tensorType, tensorLayout};

            pLogger->log(1, "Loading plugins");
            pPluginLoader = make_refobject<DynamicPluginLoader>().as<IPluginLoader>();
            std::uint32_t loadCount = 0;
            status = pPluginLoader->loadDirectory(pluginsFolder.c_str(), &loadCount);
            if (failed(status) || (0 == loadCount))
            {
                pLogger->logError(0, "Unable to load plugins");
                return;
            }
            pRegistry = make_refobject<ClassRegistry>().as<IClassRegistry>();
            status = pRegistry->importPluginLoader(pPluginLoader.get());
            if (failed(status))
            {
                pLogger->logError(0, "Unable to initialize class registry");
                return;
            }

            pLogger->log(1, "Creating network classes");
            pRegistry->createObject(pNetwork.put_refobject(), NVNEURAL_INETWORK_OBJECTCLASS);
            if (!pNetwork)
            {
                pLogger->logError(0, "Unable to create network object");
                return;
            }

            status = pNetwork->setClassRegistry(pRegistry.get());
            if (failed(status))
            {
                pLogger->logError(0, "Unable to set class registry to network object");
                return;
            }

            status = pNetwork->setDefaultTensorFormat(tensorFormat);
            if (failed(status))
            {
                pLogger->logError(0, "Unable to set tensor format");
                return;
            }
            std::string weightsPath = baseWeightsPath;
            if (guideAlbedo && guideNormal)
                weightsPath += "rt_hdr_alb_nrm";
            else if (guideAlbedo)
                weightsPath += "rt_hdr_alb";
            else
                weightsPath += "rt_hdr";
            pWeightsLoader = make_refobject<FilesystemWeightsLoader>(weightsPath).as<IWeightsLoader>();
            status = pNetwork->setWeightsLoader(pWeightsLoader.get());
            if (failed(status))
            {
                pLogger->logError(0, "Unable to register weights loader");
                return;
            }

            pLogger->log(1, "Initializing CUDA");
            pRegistry->createObject(pCudaBackend.put_refobject(), NVNEURAL_INETWORKBACKENDCUDA_OBJECTCLASS);
            if (!pCudaBackend)
            {
                pLogger->logError(0, "Unable to create CUDA backend");
                return;
            }
            status = pCudaBackend->initializeFromDeviceOrdinal(cudaDeviceIndex);
            if (failed(status))
            {
                pLogger->logError(0, "Unable to initialize CUDA");
                return;
            }
            status = pNetwork->attachBackend(pCudaBackend.get());
            if (failed(status))
            {
                pLogger->logError(0, "Unable to attach CUDA backend to network");
                return;
            }
            status = pNetwork->setDefaultBackendId(pCudaBackend->id());
            if (failed(status))
            {
                pLogger->logError(0, "Unable to set CUDA backend as default");
                return;
            }

            pLogger->log(1, "Creating network");
            status = BuildNetwork_OIDN(pNetwork, pRegistry.get());
            if (failed(status))
            {
                pLogger->logError(0, "Unable to construct network");
                return;
            }

            const auto networkBackendCuda = pCudaBackend.as<INetworkBackendCuda>();
            stream_ = networkBackendCuda->getCudaStream();
        }

        ~OidnDenoiserImpl()
        {
            if (pInputData_)
            {
                cudaFree(pInputData_);
            }

            pNetwork = nullptr;
            pRegistry = nullptr;
            pLogger = nullptr;
            pCudaBackend = nullptr;
            pWeightsLoader = nullptr;
            pPluginLoader = nullptr;
        }

        static TensorFormat GetTensorFormatForImage(const Image2D &image)
        {
            const TensorDataType type = image.dataType == DataType::Half ? TensorDataType::Half : TensorDataType::Float;
            const TensorDataLayout layout = image.dataLayout == DataLayout::NHWC ? TensorDataLayout::Nhwc : TensorDataLayout::Nchw;
            return {type, layout};
        }

        static TensorDimension GetTensorDimensionForImage(const Image2D &image)
        {
            return {1, image.numChannels, image.height, image.width};
        }

        bool UpdateDataForInputLayer(const std::string &layerName, const Image2D &colorImage, const Image2D *albedoImage, const Image2D *normalImage)
        {
            nvneural::ILayer *const pLayer_input = pNetwork->getLayerByName(layerName.c_str());
            const auto pInputLayer_input = nvneural::RefPtr<nvneural::ILayer>::fromPointer(pLayer_input).as<nvneural::ICudaInputLayer>();
            if (!pInputLayer_input)
            {
                pLogger->logError(0, "%s: Not an input layer", pLayer_input->name());
                return false;
            }

            // TODO: read tiling parameters from input layer
            size_t tiling = 16;
            size_t inputWidth = DivideRoundingUp(colorImage.width, tiling) * tiling;
            size_t inputHeight = DivideRoundingUp(colorImage.height, tiling) * tiling;
            size_t inputChannels = 3;
            if (albedoImage)
                inputChannels += 3;
            if (normalImage)
                inputChannels += 3;

            TensorFormat layerFormat = pLayer_input->tensorFormat();
            TensorDimension layerDimension = {1, inputChannels, inputHeight, inputWidth};
            TensorDimension layerImageDimension = {1, 3, inputHeight, inputWidth};
            size_t dataSizeInBytes = dataSize(layerFormat.elementType) * layerDimension.elementCount();
            size_t dataOffsetPerImage = dataSize(layerFormat.elementType) * layerImageDimension.elementCount();

            if (inputDataSize_ != dataSizeInBytes)
            {
                if (pInputData_)
                {
                    cudaFreeAsync(pInputData_, stream_);
                }

                cudaMallocAsync((void **)&pInputData_, dataSizeInBytes, stream_);
                cudaMemsetAsync(pInputData_, 0, dataSizeInBytes, stream_);
                inputDataSize_ = dataSizeInBytes;
            }

            CopyImage(stream_,
                      pInputData_, layerFormat, layerImageDimension,
                      colorImage.data, GetTensorFormatForImage(colorImage), GetTensorDimensionForImage(colorImage),
                      ETransferFunction::PU);

            if (albedoImage)
            {
                CopyImage(stream_,
                          (uint8_t*)pInputData_ + dataOffsetPerImage, layerFormat, layerImageDimension,
                          albedoImage->data, GetTensorFormatForImage(*albedoImage), GetTensorDimensionForImage(*albedoImage));
            }

            if (normalImage)
            {
                CopyImage(stream_,
                          (uint8_t *)pInputData_ + dataOffsetPerImage * 2, layerFormat, layerImageDimension,
                          normalImage->data, GetTensorFormatForImage(*normalImage), GetTensorDimensionForImage(*normalImage),
                          ETransferFunction::EncodeNormal);
            }

            const auto loadStatus = pInputLayer_input->copyCudaTensorAsync(pInputData_, layerDimension);
            if (nvneural::failed(loadStatus))
            {
                pLogger->logError(0, "%s: Unable to load file", pLayer_input->name());
                return false;
            }

            return true;
        }

        bool GetDataFromOutputLayer(const std::string &layerName, const Image2D &outputImage)
        {
            const auto pOutputLayer = nvneural::RefPtr<nvneural::ILayer>::fromPointer(pNetwork->getLayerByName(layerName.c_str()));

            void *pOutputBuffer;
            const auto status = pOutputLayer->getConstData((const void **)&pOutputBuffer, pOutputLayer->tensorFormat(), pOutputLayer.get());
            if (failed(status))
            {
                pLogger->logError(0, "failed to get data from output layer");
                return false;
            }

            CopyImage(stream_,
                      outputImage.data, GetTensorFormatForImage(outputImage), GetTensorDimensionForImage(outputImage),
                      pOutputBuffer, pOutputLayer->tensorFormat(), pOutputLayer->dimensions(),
                      ETransferFunction::PUInverse);

            return true;
        }

        void Denoise(const Image2D &colorImage, const Image2D* albedoImage, const Image2D* normalImage, const Image2D &outputImage)
        {
            // Assign input for layer 'input'
            if (!UpdateDataForInputLayer(layerName_input, colorImage, albedoImage, normalImage))
            {
                pLogger->logError(0, "Failed to update input layer.");
                return;
            }

            for (int passIndex = 0; passIndex < passCount; ++passIndex)
            {
                // Mark all input layers as 'affected' so each inference pass is meaningful
                pNetwork->getLayerByName(layerName_input.c_str())->setAffected(true);

                NeuralResult status = pNetwork->inference();
                if (failed(status))
                {
                    pLogger->logError(0, "Inference failed after %d previous successful passes.", passIndex);
                    return;
                }
            }

            // Save outputs
            GetDataFromOutputLayer(layerName_output, outputImage);
        }
    };

    OidnDenoiser::OidnDenoiser(bool guideAlbedo, bool guideNormal, const std::string& baseWeightsPath, const std::string& pluginsFolder, LogCallback logCallback)
    {
        impl = std::make_shared<OidnDenoiserImpl>(guideAlbedo, guideNormal, baseWeightsPath, pluginsFolder, logCallback);
    }

    CUstream OidnDenoiser::GetCudaStream() const
    {
        return impl != nullptr ? impl->stream_ : nullptr;
    }

    void OidnDenoiser::Denoise(const Image2D &colorImage, const Image2D* albedoImage, const Image2D* normalImage, const Image2D &outputImage) const
    {
        if (impl)
        {
            impl->Denoise(colorImage, albedoImage, normalImage, outputImage);
        }
    }
}
