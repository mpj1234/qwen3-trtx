#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "model.h"
#include "config.h"
#include "calibrator.h"
#include "NvInfer.h"

using namespace nvinfer1;
// static Logger gLogger(nvinfer1::ILogger::Severity::kVERBOSE);
static Logger gLogger;

void serialize_engine(std::string& wts_name, std::string& engine_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    // Create model to populate the network, then set the outputs and create an engine
    IHostMemory* serialized_engine = nullptr;
    serialized_engine = buildEngineQwen3_0_6B(builder, config, DataType::kFLOAT, wts_name);
    assert(serialized_engine);
    // Save engine to file
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // Close everything down
    delete serialized_engine;
    delete config;
    delete builder;
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                        IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

template<typename T>
std::vector<T> read_input(const std::string& input_data_path) {
    std::ifstream infile(input_data_path);
    if (!infile.is_open()) {
        std::cerr << "错误：无法打开文件 input_data.txt" << std::endl;
        return {};
    }
    std::vector<T> data;
    T value;

    while (infile >> value) {
        data.push_back(value);
    }
    infile.close();
    return data;
}

template<typename T>
void save_output(const std::string& output_data_path, const std::vector<T>& data) {
    std::ofstream outfile(output_data_path);
    if (!outfile.is_open()) {
        std::cerr << "错误：无法打开文件 output_data.txt" << std::endl;
        return;
    }
    for (const auto& value: data) {
        outfile << value << "\n";
    }
    outfile.close();
}

std::string get_format_shape(const Dims& shape) {
    std::stringstream output;
    char buffer[64];
    const char* fmts[] = {"%d", "x%d"};
    for (int i = 0; i < shape.nbDims; ++i) {
        // 如果是第一个维度，那么使用%d，否则使用x%d
        snprintf(buffer, sizeof(buffer), fmts[i != 0], shape.d[i]);
        output << buffer;
    }
    return output.str();
}

std::string dtype2string(DataType dtype) {
    switch (dtype) {
        case DataType::kFLOAT:
            return "FLOAT";
        case DataType::kHALF:
            return "HALF";
        case DataType::kINT8:
            return "INT8";
        case DataType::kINT32:
            return "INT32";
        case DataType::kBOOL:
            return "BOOL";
        case DataType::kUINT8:
            return "UINT8";
        case DataType::kFP8:
            return "FP8";
        case DataType::kBF16:
            return "BF16";
        case DataType::kINT64:
            return "INT64";
        case DataType::kINT4:
            return "INT4";
        case DataType::kFP4:
            return "FP4";
        default:
            return "UNKNOWN";
    }
}

std::vector<int32_t> build_position_ids(int seq_len) {
    std::vector<int32_t> position_ids(seq_len);
    std::iota(position_ids.begin(), position_ids.end(), 0);
    return position_ids;
}

std::vector<float> compute_default_inv_freq() {
    const int rotary_dim = static_cast<int>(static_cast<float>(kHeadDim) * kPartialRotaryFactor);
    assert(rotary_dim > 0);
    assert(rotary_dim <= kHeadDim);
    assert(rotary_dim % 2 == 0);

    std::vector<float> inv_freq(rotary_dim / 2);
    for (int i = 0; i < rotary_dim / 2; ++i) {
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
        inv_freq[i] = 1.0f / std::pow(kRopeTheta, exponent);
    }
    return inv_freq;
}

std::pair<std::vector<float>, std::vector<float>> build_default_rope(
    const std::vector<int32_t>& position_ids) {
    const int seq_len = static_cast<int>(position_ids.size());
    const int rotary_dim = static_cast<int>(static_cast<float>(kHeadDim) * kPartialRotaryFactor);
    const int half_dim = rotary_dim / 2;
    const auto inv_freq = compute_default_inv_freq();

    std::vector<float> cos(seq_len * kHeadDim, 1.0f);
    std::vector<float> sin(seq_len * kHeadDim, 0.0f);

    for (int pos = 0; pos < seq_len; ++pos) {
        const float position = static_cast<float>(position_ids[pos]);
        for (int i = 0; i < half_dim; ++i) {
            const float angle = position * inv_freq[i];
            const float cos_value = std::cos(angle) * kRopeAttentionScaling;
            const float sin_value = std::sin(angle) * kRopeAttentionScaling;
            cos[pos * kHeadDim + i] = cos_value;
            cos[pos * kHeadDim + half_dim + i] = cos_value;
            sin[pos * kHeadDim + i] = sin_value;
            sin[pos * kHeadDim + half_dim + i] = sin_value;
        }
    }

    return {cos, sin};
}

void print(const ICudaEngine* engine) {
    std::cout << "[INFO] Engine has " << engine->getNbIOTensors() << " bindings" << std::endl;
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        auto name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            auto dtype = engine->getTensorDataType(name);
            auto dims = engine->getTensorShape(name);
            std::cout << "[INFO] Input " << i << " name: " << name << " dtype: " << dtype2string(dtype) << " dims: " <<
                    get_format_shape(dims) << std::endl;
        } else if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
            auto dtype = engine->getTensorDataType(name);
            auto dims = engine->getTensorShape(name);
            std::cout << "[INFO] Output " << i << " name: " << name << " dtype: " << dtype2string(dtype) << " dims: " <<
                    get_format_shape(dims) << std::endl;
        } else {
            std::cerr << "[INFO] Unknown tensor I/O mode" << std::endl;
            assert(false);
        }
    }
}

unsigned int cumprod(const Dims& dims) {
    unsigned int sum = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        sum *= dims.d[i];
    }
    return sum;
}

void print_topk_logits(const std::vector<float>& logits, int topk) {
    std::vector<std::pair<int, float>> id_logit_pairs;
    id_logit_pairs.reserve(logits.size());
    for (int i = 0; i < static_cast<int>(logits.size()); ++i) {
        id_logit_pairs.emplace_back(i, logits[i]);
    }

    if (topk > static_cast<int>(id_logit_pairs.size())) {
        topk = static_cast<int>(id_logit_pairs.size());
    }

    std::partial_sort(
        id_logit_pairs.begin(),
        id_logit_pairs.begin() + topk,
        id_logit_pairs.end(),
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second > b.second;
        });

    std::cout << "[INFO] top-" << topk << " logits:" << std::endl;
    for (int i = 0; i < topk; ++i) {
        std::cout << "  rank=" << i
                  << " token_id=" << id_logit_pairs[i].first
                  << " logit=" << id_logit_pairs[i].second
                  << std::endl;
    }
}

std::pair<int32_t, float> select_top1(const std::vector<float>& logits) {
    assert(!logits.empty());
    auto iter = std::max_element(logits.begin(), logits.end());
    const int32_t token_id = static_cast<int32_t>(std::distance(logits.begin(), iter));
    return {token_id, *iter};
}

bool is_eos_token(int32_t token_id) {
    return std::find(kEosTokenIds.begin(), kEosTokenIds.end(), token_id) != kEosTokenIds.end();
}

std::pair<int32_t, float> sample_token(const std::vector<float>& logits, std::mt19937& rng) {
    assert(!logits.empty());

    std::vector<std::pair<int32_t, float>> candidates;
    candidates.reserve(logits.size());
    for (int32_t i = 0; i < static_cast<int32_t>(logits.size()); ++i) {
        candidates.emplace_back(i, logits[i] / kTemperature);
    }

    if (kTopK > 0 && kTopK < static_cast<int>(candidates.size())) {
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + kTopK,
            candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        candidates.resize(kTopK);
    } else {
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    float max_logit = candidates.front().second;
    std::vector<float> probs(candidates.size(), 0.0f);
    float sum = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp(candidates[i].second - max_logit);
        sum += probs[i];
    }
    for (float& prob : probs) {
        prob /= sum;
    }

    if (kTopP < 1.0f) {
        float cumulative = 0.0f;
        size_t keep_count = 0;
        for (; keep_count < probs.size(); ++keep_count) {
            cumulative += probs[keep_count];
            if (cumulative >= kTopP) {
                ++keep_count;
                break;
            }
        }
        keep_count = std::max<size_t>(1, std::min(keep_count, probs.size()));
        candidates.resize(keep_count);
        probs.resize(keep_count);
        float renorm = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float& prob : probs) {
            prob /= renorm;
        }
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    const int selected_index = dist(rng);
    const int32_t token_id = candidates[selected_index].first;
    const float selected_prob = probs[selected_index];
    return {token_id, selected_prob};
}

// (Mask 张量，有效位置为 0，无效位置为 1。会自动在 H_q 维度广播)
std::vector<float> make_causal_mask(int seq) {
    // 创建一个 seq * seq 的矩阵，初始化为 0 (默认为有效/可见)
    std::vector<float> mask(seq * seq, 0.0f);

    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            // Causal Mask 逻辑：
            // 只有当 j <= i 时（当前及之前），才是有效的。
            // 如果 j > i（未来），则是无效的，需要标记为 1。
            if (j > i) {
                mask[i * seq + j] = 1.0f; // 无效位置设为 1
            }
            // else: 保持 0.0f (有效位置)
        }
    }
    return mask;
}

class Qwen3PrefillRunner {
public:
    Qwen3PrefillRunner(const std::string& engine_path, int max_seq_len)
        : mMaxSeqLen(max_seq_len), mRng(42) {
        deserialize_engine(const_cast<std::string&>(engine_path), &mRuntime, &mEngine, &mContext);
        assert(mRuntime != nullptr);
        assert(mEngine != nullptr);
        assert(mContext != nullptr);
        print(mEngine);
        CUDA_CHECK(cudaStreamCreate(&mStream));
        allocate_buffers();
    }

    ~Qwen3PrefillRunner() {
        release_buffers();
        if (mStream != nullptr) {
            cudaStreamDestroy(mStream);
        }
        delete mContext;
        delete mEngine;
        delete mRuntime;
    }

    std::vector<float> run(const std::vector<int32_t>& input_ids) {
        assert(!input_ids.empty());
        assert(static_cast<int>(input_ids.size()) <= mMaxSeqLen);

        const int seq = static_cast<int>(input_ids.size());
        const auto position_ids = build_position_ids(seq);
        auto rope = build_default_rope(position_ids);
        auto mask = make_causal_mask(seq);

        set_shapes(seq);
        upload_inputs(input_ids, rope.first, rope.second, mask);
        bind_tensors();

        assert(mContext->enqueueV3(mStream));
        CUDA_CHECK(cudaStreamSynchronize(mStream));

        const auto output_shape = mContext->getTensorShape(kOutputTensorName);
        std::vector<float> output_data(cumprod(output_shape));
        CUDA_CHECK(cudaMemcpy(output_data.data(),
                              mOutputDevice,
                              output_data.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return output_data;
    }

    std::vector<int32_t> generate(std::vector<int32_t> input_ids) {
        assert(static_cast<int>(input_ids.size()) <= mMaxSeqLen);
        std::vector<int32_t> generated_ids;
        generated_ids.reserve(std::max(0, mMaxSeqLen - static_cast<int>(input_ids.size())));

        int step = 0;
        while (static_cast<int>(input_ids.size()) < mMaxSeqLen) {
            const auto logits = run(input_ids);
            const auto [token_id, score] = kDoSample ? sample_token(logits, mRng) : select_top1(logits);
            generated_ids.push_back(token_id);
            input_ids.push_back(token_id);
            std::cout << "[INFO] decode step=" << step
                      << " token_id=" << token_id
                      << (kDoSample ? " prob=" : " logit=") << score
                      << std::endl;
            if (is_eos_token(token_id)) {
                std::cout << "[INFO] stop on eos token at step=" << step << std::endl;
                break;
            }
            ++step;
        }

        if (static_cast<int>(input_ids.size()) >= mMaxSeqLen) {
            std::cout << "[INFO] stop on max sequence length=" << mMaxSeqLen << std::endl;
        }

        return generated_ids;
    }

private:
    void allocate_buffers() {
        const size_t max_input_ids_bytes = static_cast<size_t>(mMaxSeqLen) * sizeof(int32_t);
        const size_t max_rope_bytes = static_cast<size_t>(mMaxSeqLen) * kHeadDim * sizeof(float);
        const size_t max_mask_bytes = static_cast<size_t>(mMaxSeqLen) * mMaxSeqLen * sizeof(float);
        const size_t max_output_bytes = static_cast<size_t>(kVocabSize) * sizeof(float);
        const size_t cache_elems_per_layer =
            static_cast<size_t>(kBatchSize) * kNumKeyValueHeads * mMaxSeqLen * kHeadDim;
        mCacheElemsPerLayer = cache_elems_per_layer;
        const size_t total_cache_bytes =
            static_cast<size_t>(kNumHiddenLayers) * cache_elems_per_layer * sizeof(float);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mInputIdsDevice), max_input_ids_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mCosDevice), max_rope_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mSinDevice), max_rope_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mMaskDevice), max_mask_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mOutputDevice), max_output_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mKeyCacheBuffer), total_cache_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mValueCacheBuffer), total_cache_bytes));
        CUDA_CHECK(cudaMemset(mKeyCacheBuffer, 0, total_cache_bytes));
        CUDA_CHECK(cudaMemset(mValueCacheBuffer, 0, total_cache_bytes));
    }

    void release_buffers() {
        if (mInputIdsDevice) CUDA_CHECK(cudaFree(mInputIdsDevice));
        if (mCosDevice) CUDA_CHECK(cudaFree(mCosDevice));
        if (mSinDevice) CUDA_CHECK(cudaFree(mSinDevice));
        if (mMaskDevice) CUDA_CHECK(cudaFree(mMaskDevice));
        if (mOutputDevice) CUDA_CHECK(cudaFree(mOutputDevice));
        if (mKeyCacheBuffer) CUDA_CHECK(cudaFree(mKeyCacheBuffer));
        if (mValueCacheBuffer) CUDA_CHECK(cudaFree(mValueCacheBuffer));
    }

    void set_shapes(int seq) {
        assert(mContext->setInputShape(kInputIdsName, Dims2{kBatchSize, seq}));
        assert(mContext->setInputShape(kInputCosName, Dims3{kBatchSize, seq, kHeadDim}));
        assert(mContext->setInputShape(kInputSinName, Dims3{kBatchSize, seq, kHeadDim}));
        assert(mContext->setInputShape(kInputMaskName, Dims4{kBatchSize, 1, seq, seq}));

        for (int layer_idx = 0; layer_idx < kNumHiddenLayers; ++layer_idx) {
            const std::string key_cache_input_name = getKeyCacheInputName(layer_idx);
            const std::string value_cache_input_name = getValueCacheInputName(layer_idx);
            assert(mContext->setInputShape(key_cache_input_name.c_str(),
                                           Dims4{kBatchSize, kNumKeyValueHeads, 0, kHeadDim}));
            assert(mContext->setInputShape(value_cache_input_name.c_str(),
                                           Dims4{kBatchSize, kNumKeyValueHeads, 0, kHeadDim}));
        }

        const auto output_shape = mContext->getTensorShape(kOutputTensorName);
        std::cout << "[INFO] output shape: ";
        for (int i = 0; i < output_shape.nbDims; ++i) {
            std::cout << output_shape.d[i] << " ";
        }
        std::cout << std::endl;
    }

    void upload_inputs(const std::vector<int32_t>& input_ids,
                       const std::vector<float>& cos,
                       const std::vector<float>& sin,
                       const std::vector<float>& mask) {
        CUDA_CHECK(cudaMemcpy(mInputIdsDevice,
                              input_ids.data(),
                              input_ids.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mCosDevice,
                              cos.data(),
                              cos.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mSinDevice,
                              sin.data(),
                              sin.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mMaskDevice,
                              mask.data(),
                              mask.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    void bind_tensors() {
        mContext->setInputTensorAddress(kInputIdsName, mInputIdsDevice);
        mContext->setInputTensorAddress(kInputCosName, mCosDevice);
        mContext->setInputTensorAddress(kInputSinName, mSinDevice);
        mContext->setInputTensorAddress(kInputMaskName, mMaskDevice);
        mContext->setOutputTensorAddress(kOutputTensorName, mOutputDevice);

        for (int layer_idx = 0; layer_idx < kNumHiddenLayers; ++layer_idx) {
            const size_t layer_offset = static_cast<size_t>(layer_idx) * mCacheElemsPerLayer;
            float* key_cache_ptr = mKeyCacheBuffer + layer_offset;
            float* value_cache_ptr = mValueCacheBuffer + layer_offset;

            const std::string key_cache_input_name = getKeyCacheInputName(layer_idx);
            const std::string value_cache_input_name = getValueCacheInputName(layer_idx);
            const std::string key_cache_output_name = getKeyCacheOutputName(layer_idx);
            const std::string value_cache_output_name = getValueCacheOutputName(layer_idx);

            mContext->setInputTensorAddress(key_cache_input_name.c_str(), key_cache_ptr);
            mContext->setInputTensorAddress(value_cache_input_name.c_str(), value_cache_ptr);
            mContext->setOutputTensorAddress(key_cache_output_name.c_str(), key_cache_ptr);
            mContext->setOutputTensorAddress(value_cache_output_name.c_str(), value_cache_ptr);
        }
    }

private:
    int mMaxSeqLen{0};
    size_t mCacheElemsPerLayer{0};

    IRuntime* mRuntime{nullptr};
    ICudaEngine* mEngine{nullptr};
    IExecutionContext* mContext{nullptr};
    cudaStream_t mStream{nullptr};

    int32_t* mInputIdsDevice{nullptr};
    float* mCosDevice{nullptr};
    float* mSinDevice{nullptr};
    float* mMaskDevice{nullptr};
    float* mOutputDevice{nullptr};
    float* mKeyCacheBuffer{nullptr};
    float* mValueCacheBuffer{nullptr};
    std::mt19937 mRng;
};

int main() {
    std::string engine_model = "../models/test.engine";
    std::string input_data_path = "F:\\LLM\\transformers-4.57.6\\input_ids.txt";

    auto input_data = read_input<int32_t>(input_data_path);
    assert(!input_data.empty());

    Qwen3PrefillRunner runner(engine_model, kMaxSeqLen);
    auto prefill_output = runner.run(input_data);
    print_topk_logits(prefill_output, 10);
    save_output("../models/output.txt", prefill_output);

    auto generated_ids = runner.generate(input_data);
    save_output("../models/generated_ids.txt", generated_ids);

    std::vector<int32_t> full_sequence = input_data;
    full_sequence.insert(full_sequence.end(), generated_ids.begin(), generated_ids.end());
    save_output("../models/full_sequence.txt", full_sequence);
    return 0;
}
