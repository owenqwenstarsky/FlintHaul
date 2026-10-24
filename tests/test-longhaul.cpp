#include "testing.h"

#include "../src/llama-longhaul.h"
#include "../src/llama-model.h"

#include "ggml-backend.h"
#include "llama-cpp.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct longhaul_fixture {
    static constexpr size_t expert_size = 32;

    FILE * file = nullptr;
    ggml_backend_ptr backend;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * weights_a = nullptr;
    ggml_tensor * weights_b = nullptr;
    ggml_tensor * ids = nullptr;
    std::unique_ptr<llama_longhaul_cache> cache;

    longhaul_fixture(
            size_t n_slots,
            uint32_t n_experts,
            bool two_sources = false,
            uint32_t written_experts = 0) {
        file = tmpfile();
        GGML_ASSERT(file != nullptr);
        write_experts(written_experts == 0 ? n_experts : written_experts, two_sources);

        backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        GGML_ASSERT(backend);

        ggml_init_params params = {
            /*.mem_size   =*/ 16 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);

        weights_a = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, expert_size, 1, n_slots);
        if (two_sources) {
            weights_b = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, expert_size, 1, n_slots);
        }
        ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_slots);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        GGML_ASSERT(buffer);

        llama_files files;
        files.emplace_back(std::make_unique<llama_file>(file));

        std::vector<llama_model_loader::longhaul_source> sources = {
            { weights_a, 0, 0, expert_size, 0 },
        };
        if (two_sources) {
            sources.push_back({ weights_b, 0, n_experts * expert_size, expert_size, 0 });
        }
        cache = std::make_unique<llama_longhaul_cache>(
            std::move(files), std::move(sources), n_slots, n_experts, 1);
    }

    ~longhaul_fixture() {
        cache.reset();
        if (file) {
            fclose(file);
        }
    }

    void write_experts(uint32_t n_experts, bool two_sources) {
        GGML_ASSERT(fseek(file, 0, SEEK_END) == 0);
        for (int source = 0; source < (two_sources ? 2 : 1); ++source) {
            for (uint32_t expert = 0; expert < n_experts; ++expert) {
                std::vector<uint8_t> data(expert_size, uint8_t(1 + expert + source * 32));
                GGML_ASSERT(fwrite(data.data(), 1, data.size(), file) == data.size());
            }
        }
        GGML_ASSERT(fflush(file) == 0);
    }

    std::vector<int32_t> remap(std::vector<int32_t> values) {
        GGML_ASSERT(values.size() <= (size_t) ids->ne[0]);
        ggml_backend_tensor_set(ids, values.data(), 0, values.size() * sizeof(int32_t));
        ids->ne[0] = values.size();
        const bool ok = cache->remap(0, ids);
        if (ok) {
            ggml_backend_tensor_get(ids, values.data(), 0, values.size() * sizeof(int32_t));
        }
        cache->release(0);
        return ok ? values : std::vector<int32_t>{};
    }

    uint8_t slot_value(ggml_tensor * tensor, int32_t slot) {
        uint8_t value = 0;
        ggml_backend_tensor_get(tensor, &value, size_t(slot) * tensor->nb[2], 1);
        return value;
    }
};

static void test_batch_planning(testing & t) {
    longhaul_fixture fixture(3, 4);

    const auto initial = fixture.remap({0, 1, 2});
    t.assert_equal(3u, initial.size());
    t.assert_equal(3u, fixture.cache->misses());

    // Expert 0 is the oldest cache entry. A sequential miss for expert 3 used
    // to evict it before the later 0 in this same routed batch was observed.
    const auto remapped = fixture.remap({3, 0, 3});
    t.assert_equal(3u, remapped.size());
    t.assert_equal("only expert 3 is loaded", 4u, fixture.cache->misses());
    t.assert_true("duplicates map to the same slot", remapped[0] == remapped[2]);
    t.assert_true("different experts map to different slots", remapped[0] != remapped[1]);
    t.assert_equal(uint8_t(4), fixture.slot_value(fixture.weights_a, remapped[0]));
    t.assert_equal(uint8_t(1), fixture.slot_value(fixture.weights_a, remapped[1]));
}

static void test_multiple_sources(testing & t) {
    longhaul_fixture fixture(2, 4, true);

    const auto remapped = fixture.remap({2, 1});
    t.assert_equal(2u, remapped.size());
    t.assert_equal(uint64_t(4 * longhaul_fixture::expert_size), fixture.cache->bytes_read_count());
    t.assert_equal(uint8_t(3),  fixture.slot_value(fixture.weights_a, remapped[0]));
    t.assert_equal(uint8_t(35), fixture.slot_value(fixture.weights_b, remapped[0]));
    t.assert_equal(uint8_t(2),  fixture.slot_value(fixture.weights_a, remapped[1]));
    t.assert_equal(uint8_t(34), fixture.slot_value(fixture.weights_b, remapped[1]));
}

static void test_concurrent_source_reads(testing & t) {
    longhaul_fixture fixture(2, 4, true);

    for (int iteration = 0; iteration < 64; ++iteration) {
        const int32_t first = iteration % 2 == 0 ? 0 : 2;
        const int32_t second = first + 1;
        const auto remapped = fixture.remap({first, second});

        if (!t.assert_equal(2u, remapped.size())) {
            return;
        }
        t.assert_equal(
            uint8_t(1 + first),
            fixture.slot_value(fixture.weights_a, remapped[0]));
        t.assert_equal(
            uint8_t(33 + first),
            fixture.slot_value(fixture.weights_b, remapped[0]));
        t.assert_equal(
            uint8_t(1 + second),
            fixture.slot_value(fixture.weights_a, remapped[1]));
        t.assert_equal(
            uint8_t(33 + second),
            fixture.slot_value(fixture.weights_b, remapped[1]));
    }
}

static void test_invalid_ids(testing & t) {
    longhaul_fixture fixture(2, 4);

    const auto remapped = fixture.remap({0, 4});
    t.assert_true(remapped.empty());
    t.assert_true(fixture.cache->failed());
    t.assert_equal(0u, fixture.cache->misses());
}

static void test_read_failure_recovery(testing & t) {
    longhaul_fixture fixture(1, 4, false, 3);

    const auto failed = fixture.remap({3});
    t.assert_true(failed.empty());
    t.assert_true(fixture.cache->failed());
    t.assert_equal(0u, fixture.cache->misses());

    fixture.write_experts(1, false);
    const auto remapped = fixture.remap({3});
    t.assert_equal(1u, remapped.size());
    t.assert_equal(1u, fixture.cache->misses());
    t.assert_equal(uint8_t(1), fixture.slot_value(fixture.weights_a, remapped[0]));
}

struct cpu_decode_result {
    std::vector<float> logits;
    uint32_t capacity = 0;
    uint64_t misses = 0;
    uint64_t bytes_read = 0;
    bool expert_buffer_is_host = false;
};

static cpu_decode_result decode_model(
        const std::string & path,
        llama_load_mode load_mode,
        uint64_t cache_bytes,
        int32_t n_gpu_layers,
        bool incremental = false) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.load_mode = load_mode;
    model_params.longhaul_cache_bytes = cache_bytes;
    model_params.use_extra_bufts = true;

    llama_model_ptr model(llama_model_load_from_file(path.c_str(), model_params));
    if (!model) {
        throw std::runtime_error("failed to load CPU model: " + path);
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 32;
    context_params.n_batch = 8;
    context_params.n_ubatch = 8;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;

    llama_context_ptr context(llama_init_from_model(model.get(), context_params));
    if (!context) {
        throw std::runtime_error("failed to create CPU context: " + path);
    }

    std::vector<llama_token> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    if (incremental) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            llama_batch batch = llama_batch_get_one(&tokens[i], 1);
            if (llama_decode(context.get(), batch) != 0) {
                throw std::runtime_error("failed incremental decode: " + path);
            }
        }
    } else {
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(context.get(), batch) != 0) {
            throw std::runtime_error("failed batched decode: " + path);
        }
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const float * logits = llama_get_logits_ith(context.get(), -1);
    if (!logits) {
        throw std::runtime_error("CPU model produced no logits: " + path);
    }

    cpu_decode_result result;
    result.logits.assign(logits, logits + n_vocab);

    if (load_mode == LLAMA_LOAD_MODE_LONGHAUL) {
        llama_longhaul_cache * cache = model->longhaul_cache();
        if (!cache) {
            throw std::runtime_error("longhaul cache was not created: " + path);
        }
        result.capacity = cache->capacity();
        result.misses = cache->misses();
        result.bytes_read = cache->bytes_read_count();

        for (const auto & item : llama_internal_get_tensor_map(model.get())) {
            const std::string & name = item.first;
            if (name.find("ffn_") != std::string::npos &&
                name.find("_exps.weight") != std::string::npos &&
                item.second->ne[2] == result.capacity) {
                result.expert_buffer_is_host = ggml_backend_buffer_is_host(item.second->buffer);
                break;
            }
        }
    }

    return result;
}

static void test_model(
        testing & t,
        const std::string & path,
        uint64_t cache_bytes,
        int32_t n_gpu_layers,
        uint32_t expected_capacity) {
    const cpu_decode_result regular =
        decode_model(path, LLAMA_LOAD_MODE_NONE, 0, n_gpu_layers);
    const cpu_decode_result incremental =
        decode_model(path, LLAMA_LOAD_MODE_NONE, 0, n_gpu_layers, true);
    const cpu_decode_result longhaul =
        decode_model(path, LLAMA_LOAD_MODE_LONGHAUL, cache_bytes, n_gpu_layers);

    t.assert_equal(expected_capacity, longhaul.capacity);
    t.assert_true("longhaul loaded at least one expert", longhaul.misses > 0);
    t.assert_true("longhaul read expert bytes", longhaul.bytes_read > 0);
    if (n_gpu_layers == 0) {
        t.assert_true(
            "streamed CPU experts use a directly writable host buffer",
            longhaul.expert_buffer_is_host);
    }
    const auto assert_logits_close = [&](const char * name, const cpu_decode_result & actual) {
        if (!t.assert_equal(regular.logits.size(), actual.logits.size())) {
            return;
        }
        double squared_error = 0.0;
        double squared_reference = 0.0;
        for (size_t i = 0; i < regular.logits.size(); ++i) {
            const double delta = double(regular.logits[i]) - double(actual.logits[i]);
            squared_error += delta * delta;
            squared_reference += double(regular.logits[i]) * double(regular.logits[i]);
        }
        const double nmse = squared_reference > 0.0
            ? squared_error / squared_reference
            : squared_error;
        t.assert_true(
            std::string(name) + " logits NMSE = " + std::to_string(nmse),
            std::isfinite(nmse) && nmse <= 1e-6);
    };

    assert_logits_close("incremental", incremental);
    assert_logits_close("longhaul", longhaul);
}

int main(int argc, char ** argv) {
    testing t;
    llama_backend_init();

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }
    if (!t.verbose) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }

    if ((argc == 4 || argc == 5) &&
            (std::string(argv[1]) == "--cpu-model" ||
             std::string(argv[1]) == "--metal-model")) {
        const std::string path = argv[2];
        const uint64_t cache_bytes = std::stoull(argv[3]);
        const uint32_t expected_capacity = argc == 5 ? std::stoul(argv[4]) : 1;
        const bool metal = std::string(argv[1]) == "--metal-model";
        t.test(metal ? "metal_model" : "cpu_model", [&](testing & current) {
            test_model(current, path, cache_bytes, metal ? 99 : 0, expected_capacity);
        });
        const int result = t.summary();
        llama_backend_free();
        return result;
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("batch_planning",  test_batch_planning);
    t.test("multiple_sources", test_multiple_sources);
    t.test("concurrent_reads", test_concurrent_source_reads);
    t.test("invalid_ids",     test_invalid_ids);
    t.test("read_failure",    test_read_failure_recovery);

    const int result = t.summary();
    llama_backend_free();
    return result;
}
