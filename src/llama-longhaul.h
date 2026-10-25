#pragma once

#include "llama-mmap.h"
#include "llama-model-loader.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct llama_longhaul_cache {
    llama_longhaul_cache(
        llama_files files,
        std::vector<llama_model_loader::longhaul_source> sources,
        size_t n_slots,
        uint32_t n_experts,
        uint32_t n_layers);
    ~llama_longhaul_cache();

    bool remap(int layer, ggml_tensor * ids);
    void release(int layer);

    uint32_t capacity() const;
    uint32_t max_ubatch(uint32_t n_expert_used) const;
    const std::string & error() const;
    bool failed() const;
    uint64_t misses() const;
    uint64_t bytes_read_count() const;

private:
    struct layer_state {
        std::vector<int32_t> expert_ids;
        std::vector<int32_t> expert_slots;
        std::vector<uint64_t> last_used;
    };

    struct io_job {
        const llama_model_loader::longhaul_source * source;
        int32_t expert_id;
        int slot;
        bool ok = false;
        std::string error;
    };

    llama_files files;
    std::vector<llama_model_loader::longhaul_source> sources;
    std::vector<std::vector<const llama_model_loader::longhaul_source *>> sources_by_layer;
    std::vector<layer_state> layers;
    size_t n_slots;
    uint32_t n_experts;
    uint64_t tick = 0;
    uint64_t n_batches = 0;
    uint64_t n_ids = 0;
    uint64_t n_unique = 0;
    uint64_t n_duplicates = 0;
    uint64_t n_hits = 0;
    uint64_t n_misses = 0;
    uint64_t bytes_read = 0;
    uint64_t io_wall_us = 0;
    uint64_t remap_us = 0;
    std::mutex mutex;
    bool locked = false;
    int locked_layer = -1;
    std::string last_error;

    std::vector<uint8_t> requested;
    std::vector<int32_t> requested_experts;
    std::vector<int32_t> missing_experts;
    std::vector<int32_t> available_slots;
    std::vector<std::pair<int32_t, int32_t>> load_plan;
    std::vector<int32_t> id_buffer;
    std::vector<uint8_t> read_buffer;

    std::vector<std::thread> io_workers;
    std::vector<io_job> io_jobs;
    std::deque<io_job *> io_queue;
    std::mutex io_mutex;
    std::condition_variable io_ready;
    std::condition_variable io_done;
    size_t io_pending = 0;
    bool io_stopping = false;

    bool source_is_direct(const llama_model_loader::longhaul_source & source) const;
    bool load_plan_sources();
    void invalidate_plan(int layer);
    void io_worker();
};
