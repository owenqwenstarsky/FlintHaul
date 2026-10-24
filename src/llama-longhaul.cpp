#include "llama-longhaul.h"

#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <inttypes.h>

llama_longhaul_cache::llama_longhaul_cache(
        llama_files files,
        std::vector<llama_model_loader::longhaul_source> sources,
        size_t n_slots,
        uint32_t n_experts,
        uint32_t n_layers) :
    files(std::move(files)),
    sources(std::move(sources)),
    sources_by_layer(n_layers),
    layers(n_layers),
    n_slots(n_slots),
    n_experts(n_experts) {
    for (auto & file : this->files) {
        file->set_no_cache();
    }
    for (const auto & source : this->sources) {
        sources_by_layer.at(source.layer).push_back(&source);
    }
    for (auto & layer : layers) {
        layer.expert_ids.assign(n_slots, -1);
        layer.expert_slots.assign(n_experts, -1);
        layer.last_used.assign(n_slots, 0);
    }

    requested.resize(n_experts);
    requested_experts.reserve(n_slots);
    missing_experts.reserve(n_slots);
    available_slots.reserve(n_slots);
    load_plan.reserve(n_slots);
    id_buffer.reserve(n_slots);

    const unsigned int n_threads = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    io_workers.reserve(n_threads);
    for (unsigned int i = 0; i < n_threads; ++i) {
        io_workers.emplace_back(&llama_longhaul_cache::io_worker, this);
    }
}

llama_longhaul_cache::~llama_longhaul_cache() {
    {
        std::lock_guard<std::mutex> lock(io_mutex);
        io_stopping = true;
    }
    io_ready.notify_all();
    for (auto & worker : io_workers) {
        worker.join();
    }

    LLAMA_LOG_INFO(
        "%s: batches = %" PRIu64 ", ids = %" PRIu64 ", unique = %" PRIu64
        ", duplicates = %" PRIu64 ", hits = %" PRIu64 ", misses = %" PRIu64
        ", read = %.2f MiB, io = %.2f ms, remap = %.2f ms\n",
        __func__, n_batches, n_ids, n_unique, n_duplicates, n_hits, n_misses,
        bytes_read / 1024.0 / 1024.0, io_wall_us / 1000.0, remap_us / 1000.0);
}

bool llama_longhaul_cache::source_is_direct(
        const llama_model_loader::longhaul_source & source) const {
    if (ggml_backend_buffer_is_host(source.tensor->buffer)) {
        return true;
    }

    const char * name = ggml_backend_buft_name(ggml_backend_buffer_get_type(source.tensor->buffer));
    return name != nullptr &&
        std::strncmp(name, "MTL", 3) == 0 &&
        std::strstr(name, "_Private") == nullptr;
}

void llama_longhaul_cache::io_worker() {
    while (true) {
        io_job * job = nullptr;
        {
            std::unique_lock<std::mutex> lock(io_mutex);
            io_ready.wait(lock, [&] { return io_stopping || !io_queue.empty(); });
            if (io_stopping && io_queue.empty()) {
                return;
            }
            job = io_queue.front();
            io_queue.pop_front();
        }

        try {
            const auto & source = *job->source;
            void * destination =
                static_cast<uint8_t *>(source.tensor->data) + size_t(job->slot) * source.tensor->nb[2];
            files.at(source.file_idx)->read_at(
                destination, source.expert_size,
                source.offset + size_t(job->expert_id) * source.expert_size);
            job->ok = true;
        } catch (const std::exception & e) {
            job->error = e.what();
        }

        {
            std::lock_guard<std::mutex> lock(io_mutex);
            --io_pending;
        }
        io_done.notify_one();
    }
}

bool llama_longhaul_cache::load_plan_sources() {
    const size_t n_direct_sources = std::count_if(
        sources_by_layer.at(locked_layer).begin(),
        sources_by_layer.at(locked_layer).end(),
        [&](const auto * source) { return source_is_direct(*source); });
    const size_t n_jobs = load_plan.size() * n_direct_sources;
    io_jobs.clear();
    io_jobs.reserve(n_jobs);

    for (const auto & item : load_plan) {
        for (const auto * source : sources_by_layer.at(locked_layer)) {
            if (source_is_direct(*source)) {
                io_jobs.push_back({ source, item.first, item.second, false, {} });
            }
        }
    }

    if (!io_jobs.empty()) {
        std::lock_guard<std::mutex> lock(io_mutex);
        io_pending = io_jobs.size();
        for (auto & job : io_jobs) {
            io_queue.push_back(&job);
        }
        io_ready.notify_all();
    }

    bool staged_ok = true;
    for (const auto & item : load_plan) {
        const int32_t expert_id = item.first;
        const int slot = item.second;
        for (const auto * source : sources_by_layer.at(locked_layer)) {
            if (source_is_direct(*source)) {
                continue;
            }

            read_buffer.resize(source->expert_size);
            try {
                files.at(source->file_idx)->read_at(
                    read_buffer.data(), read_buffer.size(),
                    source->offset + size_t(expert_id) * source->expert_size);
                ggml_backend_tensor_set(
                    source->tensor, read_buffer.data(),
                    size_t(slot) * source->tensor->nb[2], read_buffer.size());
                bytes_read += read_buffer.size();
            } catch (const std::exception & e) {
                last_error = e.what();
                staged_ok = false;
                break;
            }
        }
        if (!staged_ok) {
            break;
        }
    }

    if (!io_jobs.empty()) {
        std::unique_lock<std::mutex> lock(io_mutex);
        io_done.wait(lock, [&] { return io_pending == 0; });
    }

    for (const auto & job : io_jobs) {
        if (!job.ok) {
            last_error = job.error;
            return false;
        }
        bytes_read += job.source->expert_size;
    }
    return staged_ok;
}

void llama_longhaul_cache::invalidate_plan(int layer) {
    auto & state = layers.at(layer);
    for (const auto & item : load_plan) {
        const int slot = item.second;
        const int32_t old_expert = state.expert_ids.at(slot);
        if (old_expert >= 0) {
            state.expert_slots.at(old_expert) = -1;
        }
        state.expert_ids.at(slot) = -1;
        state.last_used.at(slot) = 0;
    }
}

bool llama_longhaul_cache::remap(int layer, ggml_tensor * ids) {
    const int64_t t_start_us = ggml_time_us();
    last_error.clear();
    mutex.lock();
    locked = true;
    locked_layer = layer;

    const size_t n_values = ggml_nbytes(ids) / sizeof(int32_t);
    id_buffer.resize(n_values);
    ggml_backend_tensor_get(ids, id_buffer.data(), 0, ggml_nbytes(ids));

    std::fill(requested.begin(), requested.end(), 0);
    requested_experts.clear();
    missing_experts.clear();
    available_slots.clear();
    load_plan.clear();

    auto & state = layers.at(layer);
    for (const int32_t id : id_buffer) {
        if (id < 0 || id >= (int32_t) n_experts) {
            last_error = "router produced an out-of-range expert ID";
            release(layer);
            return false;
        }
        if (!requested.at(id)) {
            requested.at(id) = 1;
            requested_experts.push_back(id);
            if (state.expert_slots.at(id) < 0) {
                missing_experts.push_back(id);
            }
        }
    }

    if (requested_experts.size() > n_slots) {
        last_error = "router requested more unique experts than the longhaul cache can hold";
        release(layer);
        return false;
    }

    for (size_t slot = 0; slot < n_slots; ++slot) {
        const int32_t expert_id = state.expert_ids.at(slot);
        if (expert_id < 0 || !requested.at(expert_id)) {
            available_slots.push_back(slot);
        }
    }
    std::stable_sort(available_slots.begin(), available_slots.end(), [&](int32_t a, int32_t b) {
        const bool a_empty = state.expert_ids.at(a) < 0;
        const bool b_empty = state.expert_ids.at(b) < 0;
        if (a_empty != b_empty) {
            return a_empty;
        }
        return state.last_used.at(a) < state.last_used.at(b);
    });

    if (missing_experts.size() > available_slots.size()) {
        last_error = "longhaul could not reserve enough cache slots for the routed expert batch";
        release(layer);
        return false;
    }
    for (size_t i = 0; i < missing_experts.size(); ++i) {
        load_plan.emplace_back(missing_experts.at(i), available_slots.at(i));
    }

    const int64_t io_start_us = ggml_time_us();
    if (!load_plan_sources()) {
        invalidate_plan(layer);
        release(layer);
        return false;
    }
    io_wall_us += ggml_time_us() - io_start_us;

    for (const auto & item : load_plan) {
        const int32_t expert_id = item.first;
        const int slot = item.second;
        const int32_t old_expert = state.expert_ids.at(slot);
        if (old_expert >= 0) {
            state.expert_slots.at(old_expert) = -1;
        }
        state.expert_ids.at(slot) = expert_id;
        state.expert_slots.at(expert_id) = slot;
    }

    for (int32_t & id : id_buffer) {
        const int slot = state.expert_slots.at(id);
        GGML_ASSERT(slot >= 0);
        state.last_used.at(slot) = ++tick;
        id = slot;
    }
    ggml_backend_tensor_set(ids, id_buffer.data(), 0, ggml_nbytes(ids));

    ++n_batches;
    n_ids += n_values;
    n_unique += requested_experts.size();
    n_duplicates += n_values - requested_experts.size();
    n_misses += missing_experts.size();
    n_hits += n_values - missing_experts.size();
    remap_us += ggml_time_us() - t_start_us;
    return true;
}

void llama_longhaul_cache::release(int layer) {
    if (locked && (locked_layer == layer || layer < 0)) {
        locked = false;
        locked_layer = -1;
        mutex.unlock();
    }
}

uint32_t llama_longhaul_cache::capacity() const {
    return n_slots;
}

uint32_t llama_longhaul_cache::max_ubatch(uint32_t n_expert_used) const {
    return std::max<uint32_t>(1, n_slots / n_expert_used);
}

const std::string & llama_longhaul_cache::error() const {
    return last_error;
}

bool llama_longhaul_cache::failed() const {
    return !last_error.empty();
}

uint64_t llama_longhaul_cache::misses() const {
    return n_misses;
}

uint64_t llama_longhaul_cache::bytes_read_count() const {
    return bytes_read;
}
