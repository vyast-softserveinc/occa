#include <iostream>
#include <occa.hpp>
#include <vector>
#include "constants.h"
#include "occa/internal/utils/sys.hpp"
#include <numeric>
#include <filesystem>

std::vector<float> buildData(std::size_t size,
                             float value)
{
    std::vector<float> buffer(size);
    std::iota(buffer.begin(), buffer.end(), value);
    return buffer;
}

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

int main(int argc, const char **argv) {

    constexpr const int BLOCKS = 1024;
    const int VECTOR_SIZE = THREADS_PER_BLOCK * BLOCKS;
    const float SCALE_VALUE = 2.0f;
    occa::json deviceOpts = getDeviceOptions(argc, argv);
    auto buffer = buildData(VECTOR_SIZE, 1.0f);
    occa::device device(deviceOpts);
    occa::memory deviceBuffer = device.malloc<float>(buffer.size());

    auto intrinsicPath = std::filesystem::current_path() / "custom_intrinsics";
    occa::jsonArray intrinsics {
        intrinsicPath.string()
    };
    occa::json buildProps({
          {"transpiler-version", 3},
          {"external-intrinsics", intrinsics}
    });

  occa::kernel scaleVectorAsyncFunc = device.buildKernel("scaleVec.okl", "scaleVectorAsync", buildProps);
  occa::kernel scaleVectorSyncFunc = device.buildKernel("scaleVec.okl", "scaleVectorSync", buildProps);

  deviceBuffer.copyFrom(buffer.data(), buffer.size());
  scaleVectorAsyncFunc(deviceBuffer,
                  SCALE_VALUE,
                  static_cast<int>(deviceBuffer.size()));
  std::vector<float> checkBuffer(buffer.size(), 0.0f);
  deviceBuffer.copyTo(checkBuffer.data(), checkBuffer.size());

  constexpr const float EPSILON = 0.001f;
  for(std::size_t i = 0; i < buffer.size(); ++i) {

        bool isValid = std::abs(buffer[i] * SCALE_VALUE - checkBuffer[i]) < EPSILON;
        if(!isValid) {
            std::cout << "Validation step has failed" << std::endl;
            return 1;
        }
  }
  std::cout << "Validation step is finished" << std::endl;

  constexpr const int BENCHMARK_SHOTS = 30;
  double asyncTotal = 0.0;
  double syncTotal = 0.0;
  for(int i = 0; i < BENCHMARK_SHOTS; ++i) {
        auto t1 = occa::sys::currentTime();
        scaleVectorAsyncFunc(deviceBuffer,
                             SCALE_VALUE,
                             static_cast<int>(deviceBuffer.size()));
        auto t2 = occa::sys::currentTime();
        asyncTotal += t2 - t1;
  }

  for( int i = 0; i < BENCHMARK_SHOTS; ++i) {
        auto t1 = occa::sys::currentTime();
        scaleVectorSyncFunc(deviceBuffer,
                             SCALE_VALUE,
                             static_cast<int>(deviceBuffer.size()));
        // auto t2 = std::chrono::high_resolution_clock::now();
        auto t2 = occa::sys::currentTime();
        syncTotal += t2 - t1;
  }

  std::cout << "Average async function time: " << asyncTotal / BENCHMARK_SHOTS << std::endl;
  std::cout << "Average sync function time: " << syncTotal / BENCHMARK_SHOTS << std::endl;

  return 0;
}
