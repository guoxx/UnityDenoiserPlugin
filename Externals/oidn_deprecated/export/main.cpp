// This file was automatically generated, but should be customized/replaced to meet your needs.
#include "BuildNetwork_OIDN.h"
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
#include <string>
#include <vector>

static uint64_t currentTimeInUsec(nvneural::INetworkBackend* pBackend)
{
    pBackend->synchronize();

    using namespace std::chrono;
    return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv)
{
    using namespace nvneural;
    NeuralResult status;

    CLI::App cliApp("NvNeural inference sample");

    // Tensor format defaults are based on how ConverenceNG exported the network.
    bool fp16 = false;
    bool nhwc = false;
    int cudaDeviceIndex = 0;
    int passCount = 1;
    int verbosity = 0;
    std::string weightsPath;
    std::map<std::string, std::string> inputPaths; // layer name => file path
    std::map<std::string, std::string> outputPaths; // layer name => file path

    cliApp.add_flag("-v,--verbose", verbosity, "Increase log verbosity");
    cliApp.add_flag("--fp16,!--fp32", fp16, "use FP16 (half-precision) tensors")->capture_default_str();
    cliApp.add_flag("--nhwc,!--nchw", nhwc, "use NHWC tensor layout")->capture_default_str();
    cliApp.add_option("--device", cudaDeviceIndex, "CUDA index of desired compute device")->capture_default_str();
    cliApp.add_option("--weights", weightsPath, "path to weights tensors")->required();
    cliApp.add_option("--iterations", passCount, "number of iterations to run")->capture_default_str();
    cliApp.add_option("--input,-i,--i-input", inputPaths["input"], "tensor source for the \"input\" layer")->required();
    cliApp.add_option("--output,-o,--o-output", outputPaths["output"], "save location for output tensor of the \"output\" layer");

    CLI11_PARSE(cliApp, argc, argv);

    const auto pLogger = make_refobject<SimpleLogger>(verbosity).as<ILogger>();
    SetDefaultLogger(pLogger.get());

    const TensorDataType tensorType = fp16 ? TensorDataType::Half : TensorDataType::Float;
    const TensorDataLayout tensorLayout = nhwc ? TensorDataLayout::Nhwc : TensorDataLayout::Nchw;
    const TensorFormat tensorFormat = { tensorType, tensorLayout };

    pLogger->log(1, "Loading plugins");
    const auto pPluginLoader = make_refobject<DynamicPluginLoader>().as<IPluginLoader>();
    std::uint32_t loadCount = 0;
    status = pPluginLoader->loadDirectory(".", &loadCount);
    if (failed(status) || (0 == loadCount))
    {
        pLogger->logError(0, "Unable to load plugins");
        return 1;
    }
    const RefPtr<IClassRegistry> pRegistry = make_refobject<ClassRegistry>().as<IClassRegistry>();
    status = pRegistry->importPluginLoader(pPluginLoader.get());
    if (failed(status))
    {
        pLogger->logError(0, "Unable to initialize class registry");
        return 1;
    }

    pLogger->log(1, "Creating network classes");
    RefPtr<INetwork> pNetwork;
    pRegistry->createObject(pNetwork.put_refobject(), NVNEURAL_INETWORK_OBJECTCLASS);
    if (!pNetwork)
    {
        pLogger->logError(0, "Unable to create network object");
        return 1;
    }

    status = pNetwork->setClassRegistry(pRegistry.get());
    if (failed(status))
    {
        pLogger->logError(0, "Unable to set class registry to network object");
        return 1;
    }

    status = pNetwork->setDefaultTensorFormat(tensorFormat);
    if (failed(status))
    {
        pLogger->logError(0, "Unable to set tensor format");
        return 1;
    }
    const auto pWeightsLoader = make_refobject<FilesystemWeightsLoader>(weightsPath).as<IWeightsLoader>();
    status = pNetwork->setWeightsLoader(pWeightsLoader.get());
    if (failed(status))
    {
        pLogger->logError(0, "Unable to register weights loader");
        return 1;
    }

    pLogger->log(1, "Initializing CUDA");
    RefPtr<INetworkBackend> pCudaBackend;
    pRegistry->createObject(pCudaBackend.put_refobject(), NVNEURAL_INETWORKBACKENDCUDA_OBJECTCLASS);
    if (!pCudaBackend)
    {
        pLogger->logError(0, "Unable to create CUDA backend");
        return 1;
    }
    status = pCudaBackend->initializeFromDeviceOrdinal(cudaDeviceIndex);
    if (failed(status))
    {
        pLogger->logError(0, "Unable to initialize CUDA");
        return 1;
    }
    status = pNetwork->attachBackend(pCudaBackend.get());
    if (failed(status))
    {
        pLogger->logError(0, "Unable to attach CUDA backend to network");
        return 1;
    }
    status = pNetwork->setDefaultBackendId(pCudaBackend->id());
    if (failed(status))
    {
        pLogger->logError(0, "Unable to set CUDA backend as default");
        return 1;
    }

    pLogger->log(1, "Creating network");
    status = BuildNetwork_OIDN(pNetwork, pRegistry.get());
    if (failed(status))
    {
        pLogger->logError(0, "Unable to construct network");
        return 1;
    }

    // Assign input for layer 'input'
    {
        nvneural::ILayer* const pLayer_input = pNetwork->getLayerByName("input");
        const auto pInputLayer_input = nvneural::RefPtr<nvneural::ILayer>::fromPointer(pLayer_input).as<nvneural::IStandardInputLayer>();
        if (!pInputLayer_input)
        {
            nvneural::DefaultLogger()->logError(0, "%s: Not an input layer", pLayer_input->name());
            return 1;
        }
        const auto loadStatus = pInputLayer_input->loadDetectedFile((inputPaths["input"]).c_str());
        if (nvneural::failed(loadStatus))
        {
            nvneural::DefaultLogger()->logError(0, "%s: Unable to load file", pLayer_input->name());
            return 1;
        }

    }


    const uint64_t timeStartUsec = currentTimeInUsec(pCudaBackend.get());
    for (int passIndex = 0; passIndex < passCount; ++passIndex)
    {
        // Mark all input layers as 'affected' so each inference pass is meaningful
        pNetwork->getLayerByName("input")->setAffected(true);

        status = pNetwork->inference();
        if (failed(status))
        {
            pLogger->logError(0, "Inference failed after %d previous successful passes.", passIndex);
            return 1;
        }
    }
    const uint64_t timeEndUsec = currentTimeInUsec(pCudaBackend.get());

    const double totalTimeMsec = static_cast<double>(timeEndUsec - timeStartUsec) / 1000.0;
    const double averageTimeMsec = totalTimeMsec / passCount;
    pLogger->log(0, "%d inference passes in %.3f ms (average %.3f ms/pass, includes weights loading)", passCount, totalTimeMsec, averageTimeMsec);

    // Save outputs
    bool outputErrorsEncountered = false;
    for (const auto& layerOutputPair : outputPaths)
    {
        const std::string& layerName = layerOutputPair.first;
        const std::string& layerPath = layerOutputPair.second;
        if (layerPath.empty())
        {
            continue;
        }
        pLogger->log(1, "Saving tensor for layer %s", layerName.c_str());
        const auto pLayer = RefPtr<ILayer>::fromPointer(pNetwork->getLayerByName(layerName.c_str()));
        if (!pLayer)
        {
            pLogger->logError(0, "%s: unable to find layer in network", layerName.c_str());
            outputErrorsEncountered = true;
            continue;
        }
        if (layerPath.size() > 4 && layerPath.substr(layerPath.size() - 4) == ".npy")
        {
            const TensorFormat saveFormat{TensorDataType::Float, TensorDataLayout::Nchw};
            size_t dataBufferSize;
            const auto sizeStatus = pLayer->getCpuConstData(nullptr, 0, &dataBufferSize, saveFormat);
            if (failed(sizeStatus))
            {
                pLogger->logError(0, "%s: unable to size output tensor", layerName.c_str());
                outputErrorsEncountered = true;
                continue;
            }
            std::vector<float> dataBuffer((dataBufferSize + sizeof(float) - 1) / sizeof(float));
            size_t bytesCopied;
            const auto copyStatus = pLayer->getCpuConstData(dataBuffer.data(), dataBufferSize, &bytesCopied, saveFormat);
            if (failed(copyStatus))
            {
                pLogger->logError(0, "%s: unable to retrieve output tensor", layerName.c_str());
                outputErrorsEncountered = true;
                continue;
            }

            // Create npy dimension array
            const TensorDimension layerDimensions = pLayer->dimensions();
            std::array<unsigned long, 4> npyDimensions;
            npyDimensions[0] = static_cast<unsigned long>(layerDimensions.n);
            npyDimensions[1] = static_cast<unsigned long>(layerDimensions.c);
            npyDimensions[2] = static_cast<unsigned long>(layerDimensions.h);
            npyDimensions[3] = static_cast<unsigned long>(layerDimensions.w);

            try
            {
                npy::SaveArrayAsNumpy(layerPath, false, npyDimensions.size(), npyDimensions.data(), dataBuffer);
            }
            catch (const std::runtime_error&)
            {
                pLogger->logError(0, "%s: unable to save data", layerName.c_str());
                outputErrorsEncountered = true;
                continue;
            }
        }
        else
        {
            const auto pOutputImage = make_refobject<Image>().as<IImage>();
            const auto imageChannels = std::min(pLayer->dimensions().c, (size_t)3);
            const auto pBackend = pNetwork->getBackend(pLayer->backendId());
            if (!pBackend)
            {
                pLogger->logError(0, "%s: unable to retrieve backend", layerName.c_str());
                outputErrorsEncountered = true;
                continue;
            }

            const auto copyStatus = pBackend->saveImage(pLayer.get(), pNetwork.get(), pOutputImage.get(), ImageSpace::RgbCentered, imageChannels);
            if (failed(copyStatus))
            {
                pLogger->logError(0, "%s: unable to retrieve image tensor", layerName.c_str());
                outputErrorsEncountered = true;
                continue;
            }
            const auto saveStatus = pOutputImage.as<IFileImage>()->saveToFile(layerPath.c_str());
            if (failed(saveStatus))
            {
                pLogger->logError(0, "%s: unable to save image file", layerName.c_str());
                outputErrorsEncountered = true;
                continue;
            }
        }
    }
    if (outputErrorsEncountered)
    {
        return 1;
    }


    pNetwork->unload();
    return 0;
}


