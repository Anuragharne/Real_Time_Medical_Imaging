#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <gxf/std/allocator.hpp>
#include <gxf/std/tensor.hpp>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/inference/inference.hpp>

#include "../../reconstruction/cuda/das_kernel.h"


// ============================================================
// Streaming Mode Definition
// ============================================================

enum class StreamMode {
  DATASET_REPLAY,
  SINGLE_IMAGE,
  CUDA_RF_DAS
};


// ============================================================
// Global streaming state
// ============================================================

static StreamMode g_stream_mode = StreamMode::DATASET_REPLAY;

static std::string g_rf_bin_path;
static std::string g_params_bin_path;
static DASParams g_das_params;
static std::vector<float> g_h_rf_data;
static std::vector<float> g_h_elem_x;

static std::vector<std::string> g_stream_images;
static std::size_t g_stream_index = 0;
static std::size_t g_total_frames = 0;

static constexpr std::size_t g_warmup_frames = 10;
static std::size_t g_measured_frames = 0;

static std::chrono::steady_clock::time_point g_benchmark_start;
static bool g_benchmark_started = false;


// ============================================================
// Load RF Binary Data & Parameters
// ============================================================

void loadRFData(const std::string& rf_path, const std::string& params_path) {
  std::ifstream pFile(params_path, std::ios::binary);
  if (!pFile.is_open()) {
    throw std::runtime_error("Could not open params binary file: " + params_path);
  }
  pFile.read(reinterpret_cast<char*>(&g_das_params), sizeof(DASParams));
  pFile.close();

  std::ifstream rfFile(rf_path, std::ios::binary);
  if (!rfFile.is_open()) {
    throw std::runtime_error("Could not open RF binary file: " + rf_path);
  }
  int totalRFSamples = g_das_params.numSamples * g_das_params.numElements;
  g_h_rf_data.resize(totalRFSamples);
  rfFile.read(reinterpret_cast<char*>(g_h_rf_data.data()), totalRFSamples * sizeof(float));
  rfFile.close();

  g_h_elem_x.resize(g_das_params.numElements);
  for (int i = 0; i < g_das_params.numElements; ++i) {
    g_h_elem_x[i] = g_das_params.xMin + i * (g_das_params.xMax - g_das_params.xMin) / (g_das_params.numElements - 1);
  }
}


// ============================================================
// Process one raw RF frame through CUDA DAS Beamforming
// ============================================================

std::vector<uint8_t> processRFFrameWithCUDADAS() {
  std::vector<uint8_t> h_outBMode(g_das_params.Nz * g_das_params.Nx * 3);
  std::vector<float> h_outEnvelope(g_das_params.Nz * g_das_params.Nx);

  double warmupMs = 0.0;
  double avgMs = 0.0;

  // Single-frame execution of CUDA beamforming pipeline
  runDASBeamformingCUDA(
      g_h_rf_data.data(),
      g_h_elem_x.data(),
      g_das_params,
      h_outBMode.data(),
      h_outEnvelope.data(),
      1,
      &warmupMs,
      &avgMs
  );

  constexpr int H = 224;
  constexpr int W = 224;
  constexpr int C = 3;

  std::vector<float> nchw(C * H * W);

  // Convert HWC uint8 grayscale (224 x 224 x 3) to NCHW float32 [1, 3, 224, 224] (0-255 range)
  for (int c = 0; c < C; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const uint8_t pixel = h_outBMode[(y * W + x) * 3 + c];
        nchw[c * H * W + y * W + x] = static_cast<float>(pixel);
      }
    }
  }

  std::vector<uint8_t> raw_bytes(nchw.size() * sizeof(float));
  std::memcpy(raw_bytes.data(), nchw.data(), raw_bytes.size());
  return raw_bytes;
}


// ============================================================
// Collect ultrasound images from BUSI dataset
// ============================================================

std::vector<std::string> collectImagePaths(
    const std::string& root,
    std::size_t maximum_images) {

  const std::array<std::string, 3> class_names = {
      "benign",
      "malignant",
      "normal"
  };

  std::array<std::vector<std::string>, 3> class_images;

  for (std::size_t c = 0; c < class_names.size(); ++c) {
    const std::filesystem::path class_dir = std::filesystem::path(root) / class_names[c];

    if (!std::filesystem::exists(class_dir)) {
      throw std::runtime_error("Dataset directory does not exist: " + class_dir.string());
    }

    for (const auto& entry : std::filesystem::directory_iterator(class_dir)) {
      if (!entry.is_regular_file()) {
        continue;
      }

      const std::string filename = entry.path().filename().string();
      const std::string extension = entry.path().extension().string();

      const bool valid_extension =
          extension == ".png" || extension == ".PNG" ||
          extension == ".jpg" || extension == ".JPG" ||
          extension == ".jpeg" || extension == ".JPEG";

      if (!valid_extension) {
        continue;
      }

      if (filename.size() >= 9) {
        const std::string mask_suffix = "_mask" + extension;
        if (filename.size() >= mask_suffix.size() &&
            filename.compare(filename.size() - mask_suffix.size(), mask_suffix.size(), mask_suffix) == 0) {
          continue;
        }
      }

      class_images[c].push_back(entry.path().string());
    }

    std::sort(class_images[c].begin(), class_images[c].end());
  }

  std::vector<std::string> result;
  result.reserve(maximum_images);
  std::size_t index = 0;

  while (result.size() < maximum_images) {
    bool added_image = false;
    for (std::size_t c = 0; c < class_images.size(); ++c) {
      if (index < class_images[c].size()) {
        result.push_back(class_images[c][index]);
        added_image = true;
        if (result.size() >= maximum_images) {
          break;
        }
      }
    }
    if (!added_image) {
      break;
    }
    ++index;
  }

  return result;
}


// ============================================================
// Preprocess one image file for ResNet-18
// ============================================================

std::vector<uint8_t> preprocessImageForResNet(const std::string& image_path) {
  cv::Mat input = cv::imread(image_path, cv::IMREAD_UNCHANGED);

  if (input.empty()) {
    throw std::runtime_error("Could not read ultrasound image: " + image_path);
  }

  cv::Mat gray;
  if (input.channels() == 3) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  } else if (input.channels() == 1) {
    gray = input;
  } else {
    throw std::runtime_error("Unsupported image channel count: " + std::to_string(input.channels()));
  }

  if (gray.depth() != CV_8U) {
    throw std::runtime_error("Expected an 8-bit ultrasound image.");
  }

  cv::Mat resized;
  cv::resize(gray, resized, cv::Size(224, 224), 0, 0, cv::INTER_CUBIC);

  constexpr int H = 224;
  constexpr int W = 224;
  constexpr int C = 3;

  std::vector<float> nchw(C * H * W);

  for (int c = 0; c < C; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const uint8_t pixel = resized.at<uint8_t>(y, x);
        nchw[c * H * W + y * W + x] = static_cast<float>(pixel);
      }
    }
  }

  std::vector<uint8_t> raw_bytes(nchw.size() * sizeof(float));
  std::memcpy(raw_bytes.data(), nchw.data(), raw_bytes.size());
  return raw_bytes;
}


// ============================================================
// Custom frame source operator (Supports Image File & CUDA RF)
// ============================================================

class ImageSequenceSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ImageSequenceSourceOp)

  ImageSequenceSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("out");
    spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");
  }

  void compute(
      holoscan::InputContext&,
      holoscan::OutputContext& op_output,
      holoscan::ExecutionContext& context) override {

    if (g_stream_index >= g_stream_images.size()) {
      throw std::runtime_error("Stream index exceeded available images.");
    }

    const std::size_t frame_number = g_stream_index + 1;
    const std::string& image_path = g_stream_images[g_stream_index];

    if (frame_number == g_warmup_frames + 1 && !g_benchmark_started) {
      g_benchmark_start = std::chrono::steady_clock::now();
      g_benchmark_started = true;
    }

    std::vector<uint8_t> raw_image;
    if (g_stream_mode == StreamMode::CUDA_RF_DAS) {
      raw_image = processRFFrameWithCUDADAS();
    } else {
      raw_image = preprocessImageForResNet(image_path);
    }

    if (frame_number == 1 || frame_number == 11 || frame_number % 25 == 0 || frame_number == g_total_frames) {
      if (g_stream_mode == StreamMode::CUDA_RF_DAS) {
        std::cout << "[stream] Frame " << frame_number << "/" << g_total_frames
                  << ": CUDA RF Beamforming Sample: " << image_path << "\n";
      } else {
        std::cout << "[stream] Frame " << frame_number << "/" << g_total_frames
                  << ": Image File: " << image_path << "\n";
      }
    }

    auto gxf_context = context.context();
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(gxf_context, allocator_->gxf_cid());

    if (!allocator) {
      throw std::runtime_error("Failed to obtain GXF allocator.");
    }

    auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();
    const nvidia::gxf::Shape tensor_shape({1, 3, 224, 224});
    const auto dtype = nvidia::gxf::PrimitiveType::kFloat32;
    const uint64_t bytes_per_element = nvidia::gxf::PrimitiveTypeSize(dtype);
    const auto strides = nvidia::gxf::ComputeTrivialStrides(tensor_shape, bytes_per_element);

    const auto reshape_result = gxf_tensor->reshapeCustom(
        tensor_shape, dtype, bytes_per_element, strides, nvidia::gxf::MemoryStorageType::kHost, allocator.value());

    if (!reshape_result) {
      throw std::runtime_error("Failed to allocate CUDA-pinned host tensor.");
    }

    std::memcpy(gxf_tensor->pointer(), raw_image.data(), raw_image.size());

    auto maybe_dl_ctx = gxf_tensor->toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
      throw std::runtime_error("Failed to create DLManagedTensor context.");
    }

    auto dl_ctx = maybe_dl_ctx.value();
    auto* mem_buf_ptr = static_cast<nvidia::gxf::MemoryBuffer*>(dl_ctx->memory_ref.get());

    auto holoscan_tensor = std::make_shared<holoscan::Tensor>(dl_ctx, mem_buf_ptr);

    holoscan::TensorMap out_message;
    out_message.insert({"data", holoscan_tensor});

    op_output.emit(out_message, "out");

    ++g_stream_index;
  }

 private:
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
};


// ============================================================
// Result / benchmark operator
// ============================================================

class ResultOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ResultOp)

  ResultOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("in");
  }

  void compute(
      holoscan::InputContext& op_input,
      holoscan::OutputContext&,
      holoscan::ExecutionContext&) override {

    static std::size_t result_count = 0;

    auto maybe_message = op_input.receive<holoscan::TensorMap>("in");
    if (!maybe_message) {
      HOLOSCAN_LOG_ERROR("No inference result received.");
      return;
    }

    const auto& message = maybe_message.value();
    const auto it = message.find("softmax1000Output");

    if (it == message.end()) {
      HOLOSCAN_LOG_ERROR("Output tensor 'softmax1000Output' was not found.");
      return;
    }

    auto tensor = it->second;
    if (!tensor) {
      HOLOSCAN_LOG_ERROR("Output tensor is null.");
      return;
    }

    ++result_count;

    if (tensor->nbytes() < 3 * sizeof(float)) {
      HOLOSCAN_LOG_ERROR("Output tensor is smaller than expected.");
      return;
    }

    const float* scores = static_cast<const float*>(tensor->data());

    int predicted_index = 0;
    if (scores[1] > scores[predicted_index]) predicted_index = 1;
    if (scores[2] > scores[predicted_index]) predicted_index = 2;

    if (result_count <= g_warmup_frames || result_count % 25 == 0 || result_count == g_total_frames) {
      std::cout << "[result] Completed frame " << result_count << "/" << g_total_frames << "\n";
    }

    if (result_count == g_total_frames) {
      const auto end_time = std::chrono::steady_clock::now();
      const double elapsed_seconds = std::chrono::duration<double>(end_time - g_benchmark_start).count();

      const double effective_time_per_frame = elapsed_seconds / static_cast<double>(g_measured_frames);
      const double throughput = static_cast<double>(g_measured_frames) / elapsed_seconds;

      const char* classes[] = {"Benign", "Malignant", "Normal"};

      std::cout << "\n========================================\n";
      std::cout << "   End-to-End Holoscan Stream Result\n";
      std::cout << "========================================\n";
      std::cout << "Stream Mode           : "
                << (g_stream_mode == StreamMode::CUDA_RF_DAS ? "CUDA RF Beamforming" :
                    g_stream_mode == StreamMode::SINGLE_IMAGE ? "Single Image Replay" : "BUSI Dataset Replay")
                << "\n";
      std::cout << "Warm-up frames        : " << g_warmup_frames << "\n";
      std::cout << "Measured frames       : " << g_measured_frames << "\n";
      std::cout << "Total measured time   : " << elapsed_seconds << " s\n";
      std::cout << "Effective time/frame  : " << effective_time_per_frame * 1000.0 << " ms\n";
      std::cout << "Throughput            : " << throughput << " frames/s\n";
      std::cout << "\nFinal frame prediction\n";
      std::cout << "Benign    : " << scores[0] << "\n";
      std::cout << "Malignant : " << scores[1] << "\n";
      std::cout << "Normal    : " << scores[2] << "\n";
      std::cout << "Prediction : " << classes[predicted_index] << "\n";
      std::cout << "========================================\n\n";
    }
  }
};


// ============================================================
// Main Holoscan Application
// ============================================================

class ResNet18HoloscanApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    std::size_t measured_frames = 100;
    const char* env_frames = std::getenv("STREAM_MEASURED");
    if (env_frames != nullptr && std::string(env_frames).size() > 0) {
      measured_frames = std::stoul(env_frames);
    }

    if (measured_frames == 0) {
      throw std::runtime_error("STREAM_MEASURED must be greater than zero.");
    }

    const std::size_t requested_total = g_warmup_frames + measured_frames;

    const char* env_rf_path = std::getenv("ULTRASOUND_RF_PATH");
    const char* env_single_image = std::getenv("ULTRASOUND_IMAGE_PATH");
    const char* env_root = std::getenv("ULTRASOUND_ROOT");

    std::string dataset_root;

    if (env_rf_path != nullptr && std::string(env_rf_path).size() > 0) {
      g_stream_mode = StreamMode::CUDA_RF_DAS;
      g_rf_bin_path = env_rf_path;
      std::filesystem::path rf_p(g_rf_bin_path);
      const char* env_params_path = std::getenv("ULTRASOUND_PARAMS_PATH");
      if (env_params_path != nullptr && std::string(env_params_path).size() > 0) {
        g_params_bin_path = env_params_path;
      } else {
        g_params_bin_path = (rf_p.parent_path() / "params.bin").string();
      }

      if (!std::filesystem::exists(g_rf_bin_path)) {
        throw std::runtime_error("ULTRASOUND_RF_PATH file does not exist: " + g_rf_bin_path);
      }
      if (!std::filesystem::exists(g_params_bin_path)) {
        throw std::runtime_error("Params file does not exist: " + g_params_bin_path);
      }

      loadRFData(g_rf_bin_path, g_params_bin_path);
      g_stream_images.assign(requested_total, g_rf_bin_path);
      dataset_root = "CUDA RF DAS Reconstructed Sample: " + g_rf_bin_path;

    } else if (env_single_image != nullptr && std::string(env_single_image).size() > 0) {
      g_stream_mode = StreamMode::SINGLE_IMAGE;
      const std::string single_path = env_single_image;
      if (!std::filesystem::exists(single_path)) {
        throw std::runtime_error("ULTRASOUND_IMAGE_PATH file does not exist: " + single_path);
      }
      g_stream_images.assign(requested_total, single_path);
      dataset_root = "Single Image Sample: " + single_path;

    } else if (env_root != nullptr && std::string(env_root).size() > 0) {
      g_stream_mode = StreamMode::DATASET_REPLAY;
      dataset_root = env_root;
      g_stream_images = collectImagePaths(dataset_root, requested_total);

    } else {
      throw std::runtime_error("Neither ULTRASOUND_RF_PATH, ULTRASOUND_IMAGE_PATH, nor ULTRASOUND_ROOT is set.");
    }

    if (g_stream_images.size() < requested_total) {
      throw std::runtime_error("Not enough ultrasound inputs found. Requested " +
                               std::to_string(requested_total) + ", found " +
                               std::to_string(g_stream_images.size()) + ".");
    }

    g_total_frames = g_stream_images.size();
    g_measured_frames = g_total_frames - g_warmup_frames;

    std::cout << "\n========================================\n";
    std::cout << "       Holoscan Stream Pipeline\n";
    std::cout << "========================================\n";
    std::cout << "Stream Mode          : "
              << (g_stream_mode == StreamMode::CUDA_RF_DAS ? "CUDA RF DAS Reconstruct" :
                  g_stream_mode == StreamMode::SINGLE_IMAGE ? "Single Image Replay" : "BUSI Dataset Replay")
              << "\n";
    std::cout << "Input Source         : " << dataset_root << "\n";
    std::cout << "Warm-up frames       : " << g_warmup_frames << "\n";
    std::cout << "Measured frames      : " << g_measured_frames << "\n";
    std::cout << "Total frames         : " << g_total_frames << "\n";
    std::cout << "========================================\n";

    auto allocator = make_resource<UnboundedAllocator>("inference_allocator");

    auto source = make_operator<ImageSequenceSourceOp>(
        "source",
        make_condition<CountCondition>(static_cast<int32_t>(g_total_frames)),
        Arg("allocator", allocator));

    auto inference = make_operator<ops::InferenceOp>(
        "inference",
        from_config("inference"),
        Arg("allocator", allocator));

    auto result = make_operator<ResultOp>("result");

    add_flow(source, inference, {{"out", "receivers"}});
    add_flow(inference, result, {{"transmitter", "in"}});
  }
};


// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
  auto app = holoscan::make_application<ResNet18HoloscanApp>();

  const char* env_config = std::getenv("HOLOSCAN_CONFIG_PATH");
  std::string config_path = "resnet18.yaml";

  if (argc > 1 && argv[1] != nullptr && std::strlen(argv[1]) > 0) {
    config_path = argv[1];
  } else if (env_config != nullptr && std::strlen(env_config) > 0) {
    config_path = env_config;
  }

  app->config(config_path);
  app->run();

  return 0;
}
