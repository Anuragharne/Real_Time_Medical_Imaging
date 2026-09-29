#include <iostream>
#include <string>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp>


// ============================================================
// Result operator
// Receives the output tensor from InferenceOp and prints
// the three ResNet-18 classification scores.
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

    auto maybe_message =
        op_input.receive<holoscan::TensorMap>("in");

    if (!maybe_message) {
      HOLOSCAN_LOG_ERROR(
          "No inference result received.");
      return;
    }

    const auto& message = maybe_message.value();

    // Our ONNX output tensor name.
    auto it = message.find("softmax1000Output");

    if (it == message.end()) {
      HOLOSCAN_LOG_ERROR(
          "Output tensor 'softmax1000Output' was not found.");
      return;
    }

    auto tensor = it->second;

    if (!tensor) {
      HOLOSCAN_LOG_ERROR(
          "Output tensor is null.");
      return;
    }

// ----------------------------------------------------------
    // Print tensor shape
    // ----------------------------------------------------------

    std::cout << "\n========================================\n";
    std::cout << "      ResNet-18 Holoscan Result\n";
    std::cout << "========================================\n";

    std::cout << "Output tensor shape: ";

    for (auto dim : tensor->shape()) {
      std::cout << dim << " ";
    }

    std::cout << "\n";

    // ----------------------------------------------------------
    // Make sure the output contains 3 float scores
    // ----------------------------------------------------------

    if (tensor->nbytes() < 3 * sizeof(float)) {
      HOLOSCAN_LOG_ERROR(
          "Output tensor is smaller than expected.");
      return;
    }

    const float* scores =
        static_cast<const float*>(tensor->data());

    // ----------------------------------------------------------
    // Print scores
    // ----------------------------------------------------------

    std::cout << "Benign    : " << scores[0] << "\n";
    std::cout << "Malignant : " << scores[1] << "\n";
    std::cout << "Normal    : " << scores[2] << "\n";

    // ----------------------------------------------------------
    // Determine predicted class
    // ----------------------------------------------------------

    int predicted_index = 0;

    if (scores[1] > scores[predicted_index]) {
      predicted_index = 1;
    }

    if (scores[2] > scores[predicted_index]) {
      predicted_index = 2;
    }

    const char* classes[] = {
        "Benign",
        "Malignant",
        "Normal"
    };

    std::cout << "Prediction : "
              << classes[predicted_index]
              << "\n";

    std::cout << "========================================\n\n";
  }
};

// ============================================================
// Main Holoscan application
// ============================================================

class ResNet18HoloscanApp : public holoscan::Application {
 public:

  void compose() override {

    using namespace holoscan;

    // ----------------------------------------------------------
    // Allocator
    //
    // InferenceOp requires an allocator for output tensors.
    // UnboundedAllocator is appropriate for this first prototype.
    // ----------------------------------------------------------

    auto allocator =
        make_resource<UnboundedAllocator>(
            "inference_allocator");


    // ----------------------------------------------------------
    // Source
    //
    // Generates ONE test tensor:
    //
    // [1, 3, 224, 224]
    //
    // This is only a controlled integration test.
    // The tensor currently contains zeros.
    // ----------------------------------------------------------

    auto source =
        make_operator<ops::PingTensorTxOp>(
            "source",

            make_condition<CountCondition>(1),

            Arg("batch_size", 1),
            Arg("rows", 3),
            Arg("columns", 224),
            Arg("channels", 224),

            Arg("data_type", std::string("float")),
            Arg("storage_type", std::string("host")),
            Arg("tensor_name", std::string("data")),

            // Give the source the same allocator.
            Arg("allocator", allocator)
        );


    // ----------------------------------------------------------
    // ResNet-18 inference
    //
    // Configuration comes from resnet18.yaml.
    // ----------------------------------------------------------

    auto inference =
        make_operator<ops::InferenceOp>(
            "inference",

            from_config("inference"),

            // REQUIRED by InferenceOp.
            Arg("allocator", allocator)
        );


    // ----------------------------------------------------------
    // Result printer
    // ----------------------------------------------------------

    auto result =
        make_operator<ResultOp>("result");


    // ----------------------------------------------------------
    // Connect:
    //
    // source → inference → result
    // ----------------------------------------------------------

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

  // Default configuration file.
  std::string config_path =
      "resnet18.yaml";

  // Optional command-line configuration file.
  if (argc >= 2) {
    config_path = argv[1];
  }

  app->config(config_path);

  app->run();

  return 0;
}
