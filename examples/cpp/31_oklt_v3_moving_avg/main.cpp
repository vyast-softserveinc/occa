#include <iostream>
#include <occa.hpp>
#include <vector>
#include "constants.h"

bool starts_with(const std::string &str, const std::string &substring) {
    return str.rfind(substring, 0) == 0;
}

occa::json getDeviceOptions(int argc, const char **argv) {
    for(int i  = 0; i < argc; ++i) {
        std::string argument(argv[i]);
        if((starts_with(argument,"-d") || starts_with(argument, "--device")) && i + 1 < argc)
        {
            std::string value(argv[i + 1]);
            return occa::json::parse(value);
        }
    }
    return occa::json::parse("{mode: 'Serial'}");
}

std::vector<float> buildMovingAverageData(std::size_t size,
                                          float initialValue,
                                          float fluctuation)
{
    std::vector<float> buffer(size);
    float currentValue = initialValue;
    float longIncrement = 1.0f;
    float fluctuationIncrement = fluctuation;
    for(std::size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = currentValue;
        fluctuationIncrement = -fluctuationIncrement;
        if(i % WINDOW_SIZE == 0) {
            longIncrement = -longIncrement;
        }
        currentValue += longIncrement + fluctuationIncrement;
    }
    return buffer;
}

std::vector<float> goldMovingAverage(const std::vector<float> &hostVector) {
    std::vector<float> result(hostVector.size() - WINDOW_SIZE);
    for(std::size_t i = 0; i < result.size(); ++i) {
        float value = 0.0f;
        for(std::size_t j = 0; j < WINDOW_SIZE; ++j) {
            value += hostVector[i + j];
        }
        result[i] = value / WINDOW_SIZE;
    }
    return result;
}

int runMovingAverageTest(occa::device &device, occa::json &buildProps) {
  auto inputHostBuffer = buildMovingAverageData(THREADS_PER_BLOCK * WINDOW_SIZE + WINDOW_SIZE, 10.0f, 4.0f);
  std::vector<float> outputHostBuffer(inputHostBuffer.size() - WINDOW_SIZE);

  occa::memory deviceInput = device.malloc<float>(inputHostBuffer.size());
  occa::memory deviceOutput = device.malloc<float>(outputHostBuffer.size());

  occa::kernel movingAverageKernel = device.buildKernel("movingAverage.okl", "movingAverage32f", buildProps);

  deviceInput.copyFrom(inputHostBuffer.data(), inputHostBuffer.size());

  movingAverageKernel(deviceInput,
                      static_cast<int>(inputHostBuffer.size()),
                      deviceOutput,
                      static_cast<int>(deviceOutput.size()));

  // Copy result to the host
  deviceOutput.copyTo(&outputHostBuffer[0], outputHostBuffer.size());

  auto goldValue = goldMovingAverage(inputHostBuffer);

  constexpr const float EPSILON = 0.001f;
  for(std::size_t i = 0; i < outputHostBuffer.size(); ++i) {
        bool isValid = std::abs(goldValue[i] - outputHostBuffer[i]) < EPSILON;
        if(!isValid) {
            std::cout << "Comparison with gold values has failed" << std::endl;
            return 1;
        }
  }

  std::cout << "Comparison with gold has passed" << std::endl;
  std::cout << "Moving average finished" << std::endl;

  return 0;
}

int runVectorDotTest(occa::device &device, occa::json &buildProps) {
  const std::size_t size = 1e7;
  auto vecA = std::vector<double>(size);
  auto vecB = std::vector<double>(size);
  auto vecT = std::vector<double>((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

  for (std::size_t i = 0; i < vecA.size(); i++)
    vecA[i] = i + 1, vecB[i] = size - (i + 1);

  occa::memory deviceVecA = device.malloc<double>(vecA.size());
  deviceVecA.copyFrom(vecA.data(), vecA.size());

  occa::memory deviceVecB = device.malloc<double>(vecB.size());
  deviceVecB.copyFrom(vecB.data(), vecB.size());

  occa::memory deviceVecT = device.malloc<double>(vecT.size(), 0);

  occa::kernel vectorDotKernel = device.buildKernel("vectorDot.okl", "vectorDot",
    buildProps);

  vectorDotKernel(deviceVecT, deviceVecA.size(), deviceVecA, deviceVecB);

  deviceVecT.copyTo(vecT.data(), deviceVecT.size());

  double dot = 0;
  for (std::size_t i = 0; i < vecT.size(); i++)
    dot += vecT[i];

  const double exact = (size * (size + 1.0) * (size - 1.0)) / 6;
  return (std::fabs(dot - exact)/exact > 1e-8);
}

int main(int argc, const char **argv) {

  occa::json deviceOpts = getDeviceOptions(argc, argv);
  occa::device device(deviceOpts);

  occa::json buildProps({
      {"transpiler-version", 3}
  });

  int failure = 0;
  failure |= runMovingAverageTest(device, buildProps);
  failure |= runVectorDotTest(device, buildProps);

  return failure;
}
