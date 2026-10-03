#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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


// ============================================================
// Global streaming state
// ============================================================

static std::vector<std::string> g_stream_images;

static std::size_t g_stream_index = 0;

static std::size_t g_total_frames = 0;

static constexpr std::size_t g_warmup_frames = 10;

static std::size_t g_measured_frames = 0;

static std::chrono::steady_clock::time_point
    g_benchmark_start;

static bool g_benchmark_started = false;


// ============================================================
// Collect ultrasound images from BUSI dataset
//
// Expected structure:
//
// Dataset_BUSI_with_GT/
// ├── benign/
// ├── malignant/
// └── normal/
//
// Important:
// BUSI also contains mask files such as:
//
//     benign (1)_mask.png
//
// These MUST NOT be fed to the classifier.
// ============================================================

std::vector<std::string> collectImagePaths(
    const std::string& root,
    std::size_t maximum_images) {

  const std::array<std::string, 3> class_names = {
      "benign",
      "malignant",
      "normal"
  };

  std::array<std::vector<std::string>, 3>
      class_images;


  // ----------------------------------------------------------
  // Collect original ultrasound images from every class.
  // ----------------------------------------------------------

  for (std::size_t c = 0;
       c < class_names.size();
       ++c) {

    const std::filesystem::path class_dir =
        std::filesystem::path(root)
        / class_names[c];

    if (!std::filesystem::exists(class_dir)) {

      throw std::runtime_error(
          "Dataset directory does not exist: " +
          class_dir.string());
    }


    for (const auto& entry :
         std::filesystem::directory_iterator(class_dir)) {

      if (!entry.is_regular_file()) {
        continue;
      }

      const std::string filename =
          entry.path().filename().string();

      const std::string extension =
          entry.path().extension().string();


      // ------------------------------------------------------
      // Accept standard image files only.
      // ------------------------------------------------------

      const bool valid_extension =
          extension == ".png" ||
          extension == ".PNG" ||
          extension == ".jpg" ||
          extension == ".JPG" ||
          extension == ".jpeg" ||
          extension == ".JPEG";


      if (!valid_extension) {
        continue;
      }


      // ------------------------------------------------------
      // IMPORTANT:
      //
      // Do NOT include BUSI ground-truth mask images.
      // ------------------------------------------------------

      if (filename.size() >= 9) {

        const std::string mask_suffix =
            "_mask" +
            extension;

        if (filename.size() >=
            mask_suffix.size() &&
            filename.compare(
                filename.size() -
                    mask_suffix.size(),
                mask_suffix.size(),
                mask_suffix) == 0) {

          continue;
        }
      }


      class_images[c].push_back(
          entry.path().string());
    }


    std::sort(
        class_images[c].begin(),
        class_images[c].end());
  }


  // ----------------------------------------------------------
  // Build a balanced round-robin stream:
  //
  // benign → malignant → normal
  // benign → malignant → normal
  // ...
  //
  // This is for pipeline testing only.
  // It is NOT an accuracy evaluation split.
  // ----------------------------------------------------------

  std::vector<std::string> result;

  result.reserve(maximum_images);

  std::size_t index = 0;


  while (result.size() <
         maximum_images) {

    bool added_image = false;


    for (std::size_t c = 0;
         c < class_images.size();
         ++c) {

      if (index <
          class_images[c].size()) {

        result.push_back(
            class_images[c][index]);

        added_image = true;


        if (result.size() >=
            maximum_images) {

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
// Preprocess one ultrasound image
//
// Same logical sequence as preprocessResNet.m:
//
//   1. Read image
//   2. RGB → grayscale if needed
//   3. Resize to 224 × 224
//   4. Replicate grayscale into 3 channels
//   5. Convert to NCHW float32
//
// Final tensor:
//
//   [1, 3, 224, 224]
//
// Pixel values remain in the original 0–255 range.
// ============================================================

std::vector<uint8_t> preprocessImageForResNet(
    const std::string& image_path) {

  cv::Mat input =
      cv::imread(
          image_path,
          cv::IMREAD_UNCHANGED);


  if (input.empty()) {

    throw std::runtime_error(
        "Could not read ultrasound image: " +
        image_path);
  }


  cv::Mat gray;


  // ----------------------------------------------------------
  // Match MATLAB preprocessing.
  // ----------------------------------------------------------

  if (input.channels() == 3) {

    cv::cvtColor(
        input,
        gray,
        cv::COLOR_BGR2GRAY);

  } else if (input.channels() == 1) {

    gray = input;

  } else {

    throw std::runtime_error(
        "Unsupported image channel count: " +
        std::to_string(
            input.channels()));
  }


  if (gray.depth() != CV_8U) {

    throw std::runtime_error(
        "Expected an 8-bit ultrasound image.");
  }


  // ----------------------------------------------------------
  // Resize to 224 × 224.
  // ----------------------------------------------------------

  cv::Mat resized;

  cv::resize(
      gray,
      resized,
      cv::Size(224, 224),
      0,
      0,
      cv::INTER_LINEAR);


  // ----------------------------------------------------------
  // Convert grayscale image to NCHW float32.
  //
  // Three channels are identical.
  // ----------------------------------------------------------

  constexpr int H = 224;
  constexpr int W = 224;
  constexpr int C = 3;


  std::vector<float> nchw(
      C * H * W);


  for (int c = 0;
       c < C;
       ++c) {

    for (int y = 0;
         y < H;
         ++y) {

      for (int x = 0;
           x < W;
           ++x) {

        const uint8_t pixel =
            resized.at<uint8_t>(
                y,
                x);


        nchw[
            c * H * W +
            y * W +
            x
        ] = static_cast<float>(
            pixel);
      }
    }
  }


  // ----------------------------------------------------------
  // Convert float32 data to raw bytes.
  // ----------------------------------------------------------

  std::vector<uint8_t> raw_bytes(
      nchw.size() *
      sizeof(float));


  std::memcpy(
      raw_bytes.data(),
      nchw.data(),
      raw_bytes.size());


  return raw_bytes;
}


// ============================================================
// Custom image-stream source
//
// Every compute() call:
//
//   different ultrasound image
//             ↓
//   OpenCV preprocessing
//             ↓
//   CUDA-pinned host tensor
//             ↓
//   TensorMap
//             ↓
//   Holoscan InferenceOp
// ============================================================

class ImageSequenceSourceOp
    : public holoscan::Operator {

 public:

  HOLOSCAN_OPERATOR_FORWARD_ARGS(
      ImageSequenceSourceOp)

  ImageSequenceSourceOp() = default;


  void setup(
      holoscan::OperatorSpec& spec) override {

    spec.output<
        holoscan::TensorMap>("out");


    spec.param(
        allocator_,
        "allocator",
        "Allocator",
        "Allocator used to allocate "
        "tensor output.");
  }


  void compute(
      holoscan::InputContext&,
      holoscan::OutputContext& op_output,
      holoscan::ExecutionContext& context)
      override {


    if (g_stream_index >=
        g_stream_images.size()) {

      throw std::runtime_error(
          "Stream index exceeded "
          "available images.");
    }


    const std::size_t frame_number =
        g_stream_index + 1;


    const std::string& image_path =
        g_stream_images[
            g_stream_index];


    // --------------------------------------------------------
    // Start timing when the first measured image enters the
    // pipeline.
    //
    // Frames 1-10 = warm-up
    // Frames 11-110 = measured
    // --------------------------------------------------------

    if (frame_number ==
            g_warmup_frames + 1 &&
        !g_benchmark_started) {

      g_benchmark_start =
          std::chrono::steady_clock::now();

      g_benchmark_started = true;
    }


    // --------------------------------------------------------
    // Per-frame preprocessing.
    // --------------------------------------------------------

    std::vector<uint8_t> raw_image =
        preprocessImageForResNet(
            image_path);


    // --------------------------------------------------------
    // Print selected stream progress.
    // --------------------------------------------------------

    if (frame_number == 1 ||
        frame_number == 11 ||
        frame_number % 25 == 0 ||
        frame_number == g_total_frames) {

      std::cout
          << "[stream] Frame "
          << frame_number
          << "/"
          << g_total_frames
          << ": "
          << image_path
          << "\n";
    }


    // --------------------------------------------------------
    // Get the underlying GXF allocator.
    //
    // This follows the same mechanism used by Holoscan's
    // PingTensorTxOp implementation.
    // --------------------------------------------------------

    auto gxf_context =
        context.context();


    auto allocator =
        nvidia::gxf::Handle<
            nvidia::gxf::Allocator>::Create(
                gxf_context,
                allocator_->gxf_cid());


    if (!allocator) {

      throw std::runtime_error(
          "Failed to obtain GXF allocator.");
    }


    // --------------------------------------------------------
    // Create the GXF tensor.
    // --------------------------------------------------------

    auto gxf_tensor =
        std::make_shared<
            nvidia::gxf::Tensor>();


    // --------------------------------------------------------
    // Tensor shape:
    //
    // [batch, channels, height, width]
    //
    // = [1, 3, 224, 224]
    // --------------------------------------------------------

    const nvidia::gxf::Shape tensor_shape({
        1,
        3,
        224,
        224
    });


    const auto dtype =
        nvidia::gxf::PrimitiveType::kFloat32;


    const uint64_t
        bytes_per_element =
        nvidia::gxf::PrimitiveTypeSize(
            dtype);


    const auto strides =
        nvidia::gxf::ComputeTrivialStrides(
            tensor_shape,
            bytes_per_element);


    // --------------------------------------------------------
    // Allocate CUDA-pinned host memory.
    // --------------------------------------------------------

    const auto reshape_result =
        gxf_tensor->reshapeCustom(
            tensor_shape,
            dtype,
            bytes_per_element,
            strides,
            nvidia::gxf::
                MemoryStorageType::kHost,
            allocator.value());


    if (!reshape_result) {

      throw std::runtime_error(
          "Failed to allocate "
          "CUDA-pinned host tensor.");
    }


    // --------------------------------------------------------
    // Copy preprocessed image into tensor.
    // --------------------------------------------------------

    std::memcpy(
        gxf_tensor->pointer(),
        raw_image.data(),
        raw_image.size());


    // --------------------------------------------------------
    // Convert GXF tensor to Holoscan tensor.
    // --------------------------------------------------------

    auto maybe_dl_ctx =
        gxf_tensor->
            toDLManagedTensorContext();


    if (!maybe_dl_ctx) {

      throw std::runtime_error(
          "Failed to create "
          "DLManagedTensor context.");
    }


    auto dl_ctx =
        maybe_dl_ctx.value();


    auto* mem_buf_ptr =
        static_cast<
            nvidia::gxf::MemoryBuffer*>(
                dl_ctx->memory_ref.get());


    auto holoscan_tensor =
        std::make_shared<
            holoscan::Tensor>(
                dl_ctx,
                mem_buf_ptr);


    // --------------------------------------------------------
    // Build the TensorMap.
    // --------------------------------------------------------

    holoscan::TensorMap out_message;


    out_message.insert({
        "data",
        holoscan_tensor
    });


    // --------------------------------------------------------
    // Emit the current image.
    // --------------------------------------------------------

    op_output.emit(
        out_message,
        "out");


    ++g_stream_index;
  }


 private:

  holoscan::Parameter<
      std::shared_ptr<
          holoscan::Allocator>>
      allocator_;
};


// ============================================================
// Result / benchmark operator
// ============================================================

class ResultOp
    : public holoscan::Operator {

 public:

  HOLOSCAN_OPERATOR_FORWARD_ARGS(
      ResultOp)

  ResultOp() = default;


  void setup(
      holoscan::OperatorSpec& spec) override {

    spec.input<
        holoscan::TensorMap>("in");
  }


  void compute(
      holoscan::InputContext& op_input,
      holoscan::OutputContext&,
      holoscan::ExecutionContext&)
      override {


    static std::size_t result_count =
        0;


    auto maybe_message =
        op_input.receive<
            holoscan::TensorMap>("in");


    if (!maybe_message) {

      HOLOSCAN_LOG_ERROR(
          "No inference result received.");

      return;
    }


    const auto& message =
        maybe_message.value();


    const auto it =
        message.find(
            "softmax1000Output");


    if (it == message.end()) {

      HOLOSCAN_LOG_ERROR(
          "Output tensor "
          "'softmax1000Output' "
          "was not found.");

      return;
    }


    auto tensor =
        it->second;


    if (!tensor) {

      HOLOSCAN_LOG_ERROR(
          "Output tensor is null.");

      return;
    }


    ++result_count;


    // --------------------------------------------------------
    // Validate output tensor.
    // --------------------------------------------------------

    if (tensor->nbytes() <
        3 * sizeof(float)) {

      HOLOSCAN_LOG_ERROR(
          "Output tensor is smaller "
          "than expected.");

      return;
    }


    const float* scores =
        static_cast<
            const float*>(
                tensor->data());


    // --------------------------------------------------------
    // Determine predicted class.
    // --------------------------------------------------------

    int predicted_index = 0;


    if (scores[1] >
        scores[predicted_index]) {

      predicted_index = 1;
    }


    if (scores[2] >
        scores[predicted_index]) {

      predicted_index = 2;
    }


    // --------------------------------------------------------
    // Print occasional progress.
    // --------------------------------------------------------

    if (result_count <=
            g_warmup_frames ||
        result_count % 25 == 0 ||
        result_count ==
            g_total_frames) {

      std::cout
          << "[result] Completed frame "
          << result_count
          << "/"
          << g_total_frames
          << "\n";
    }


    // --------------------------------------------------------
    // Final benchmark result.
    // --------------------------------------------------------

    if (result_count ==
        g_total_frames) {


      const auto benchmark_end =
          std::chrono::steady_clock::now();


      if (!g_benchmark_started) {

        throw std::runtime_error(
            "Benchmark timer was never started.");
      }


      const double
          elapsed_seconds =
              std::chrono::duration<double>(
                  benchmark_end -
                  g_benchmark_start)
                  .count();


      const double
          effective_time_per_frame =
              elapsed_seconds /
              static_cast<double>(
                  g_measured_frames);


      const double throughput =
          static_cast<double>(
              g_measured_frames) /
          elapsed_seconds;


      const char* classes[] = {
          "Benign",
          "Malignant",
          "Normal"
      };


      // ------------------------------------------------------
      // Print final benchmark.
      // ------------------------------------------------------

      std::cout
          << "\n========================================\n";

      std::cout
          << "   End-to-End Holoscan Stream Result\n";

      std::cout
          << "========================================\n";


      std::cout
          << "Warm-up frames        : "
          << g_warmup_frames
          << "\n";


      std::cout
          << "Measured frames       : "
          << g_measured_frames
          << "\n";


      std::cout
          << "Total measured time   : "
          << elapsed_seconds
          << " s\n";


      std::cout
          << "Effective time/frame  : "
          << effective_time_per_frame
          << " s\n";


      std::cout
          << "Effective time/frame  : "
          << effective_time_per_frame *
                 1000.0
          << " ms\n";


      std::cout
          << "Throughput            : "
          << throughput
          << " frames/s\n";


      std::cout
          << "\nFinal frame prediction\n";


      std::cout
          << "Benign    : "
          << scores[0]
          << "\n";


      std::cout
          << "Malignant : "
          << scores[1]
          << "\n";


      std::cout
          << "Normal    : "
          << scores[2]
          << "\n";


      std::cout
          << "Prediction : "
          << classes[predicted_index]
          << "\n";


      std::cout
          << "========================================\n\n";
    }
  }
};


// ============================================================
// Main Holoscan application
// ============================================================

class ResNet18HoloscanApp
    : public holoscan::Application {

 public:

  void compose() override {

    using namespace holoscan;


    // --------------------------------------------------------
    // Dataset root.
    // --------------------------------------------------------

    const char* env_root =
        std::getenv(
            "ULTRASOUND_ROOT");


    if (env_root == nullptr ||
        std::string(env_root).empty()) {

      throw std::runtime_error(
          "ULTRASOUND_ROOT environment "
          "variable is not set.");
    }


    const std::string dataset_root =
        env_root;


    // --------------------------------------------------------
    // Number of measured images.
    //
    // Default:
    //
    // 10 warm-up
    // 100 measured
    //
    // = 110 total images.
    // --------------------------------------------------------

    std::size_t measured_frames =
        100;


    const char* env_frames =
        std::getenv(
            "STREAM_MEASURED");


    if (env_frames != nullptr &&
        std::string(env_frames).size() > 0) {

      measured_frames =
          std::stoul(env_frames);
    }


    if (measured_frames == 0) {

      throw std::runtime_error(
          "STREAM_MEASURED must be "
          "greater than zero.");
    }


    const std::size_t requested_total =
        g_warmup_frames +
        measured_frames;


    // --------------------------------------------------------
    // Find real ultrasound images.
    // --------------------------------------------------------

    g_stream_images =
        collectImagePaths(
            dataset_root,
            requested_total);


    if (g_stream_images.size() <
        requested_total) {

      throw std::runtime_error(
          "Not enough ultrasound images "
          "were found. Requested " +
          std::to_string(
              requested_total) +
          ", found " +
          std::to_string(
              g_stream_images.size()) +
          ".");
    }


    g_total_frames =
        g_stream_images.size();


    g_measured_frames =
        g_total_frames -
        g_warmup_frames;


    // --------------------------------------------------------
    // Print stream information.
    // --------------------------------------------------------

    std::cout
        << "\n========================================\n";

    std::cout
        << "       Holoscan Image Stream\n";

    std::cout
        << "========================================\n";


    std::cout
        << "Dataset root         : "
        << dataset_root
        << "\n";


    std::cout
        << "Warm-up frames       : "
        << g_warmup_frames
        << "\n";


    std::cout
        << "Measured frames      : "
        << g_measured_frames
        << "\n";


    std::cout
        << "Total frames         : "
        << g_total_frames
        << "\n";


    std::cout
        << "========================================\n";


    // --------------------------------------------------------
    // Shared allocator.
    // --------------------------------------------------------

    auto allocator =
        make_resource<
            UnboundedAllocator>(
                "inference_allocator");


    // --------------------------------------------------------
    // Image source.
    // --------------------------------------------------------

    auto source =
        make_operator<
            ImageSequenceSourceOp>(
                "source",

                make_condition<
                    CountCondition>(
                        static_cast<int32_t>(
                            g_total_frames)),

                Arg(
                    "allocator",
                    allocator)
            );


    // --------------------------------------------------------
    // ResNet-18 TensorRT inference.
    // --------------------------------------------------------

    auto inference =
        make_operator<
            ops::InferenceOp>(
                "inference",

                from_config(
                    "inference"),

                Arg(
                    "allocator",
                    allocator)
            );


    // --------------------------------------------------------
    // Result.
    // --------------------------------------------------------

    auto result =
        make_operator<ResultOp>(
            "result");


    // --------------------------------------------------------
    // Graph:
    //
    // ultrasound image
    //        ↓
    // OpenCV preprocessing
    //        ↓
    // CUDA-pinned host tensor
    //        ↓
    // Holoscan
    //        ↓
    // TensorRT
    //        ↓
    // ResNet-18
    //        ↓
    // Result
    // --------------------------------------------------------

    add_flow(
        source,
        inference,
        {{"out", "receivers"}}
    );


    add_flow(
        inference,
        result,
        {{"transmitter", "in"}}
    );
  }
};


// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {

  auto app =
      holoscan::make_application<
          ResNet18HoloscanApp>();


  app->config(
      "resnet18.yaml");


  app->run();


  return 0;
}
