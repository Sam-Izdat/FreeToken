// CPU-compute MoVA (Mixture-of-Values Attention) value-expert executor.
//
// Decode ships activations to the CPU, computes the routed value experts here
// (reading bf16 v_expert weights at full RAM bandwidth), and ships the values
// back. To keep the whole decode path inside a single CUDA graph we expose
// submit/sync as host nodes via cudaLaunchHostFunc -- the callbacks only touch a
// CPU worker pool + pinned host buffers and never call any CUDA API.
//
// Architecture borrows freetoken's _cpu_moe executor (host-node submit/sync,
// persistent worker pool, bf16 GEMV microkernels with runtime ISA dispatch).
// Intentionally a focused subset: MoVA is a single bf16 GEMV per expert route
// (2560 -> 1024, no bias) + silu + router-weighted sum (no gate/up split, no
// swiglu, no down_proj, no quantization formats). The per-(token, output-col)
// unit is independent, so the run splits (tokens x col-blocks) across workers.
// No flag-handshake/coordinator: MoVA's CPU GEMV is ~ms-scale per layer, which
// dwarfs the ~30-50us host-func dispatch (the handshake only pays off when the
// CPU op is fast enough that dispatch dominates).

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <torch/extension.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#define CPU_MOVA_HAS_AFFINITY 1
#else
#define CPU_MOVA_HAS_AFFINITY 0
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define CPU_MOVA_X86 1
#else
#define CPU_MOVA_X86 0
#endif

namespace {

using bf16_t = uint16_t;

inline float bf16_to_f32(bf16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline bf16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  // round-to-nearest-even
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<bf16_t>(u >> 16);
}

inline float silu_f(float v) { return v / (1.0f + std::exp(-v)); }

// ------------------------------- dot products -------------------------------
// dot(weight[bf16], act[bf16], n) -> fp32. Selected impl is a function pointer
// chosen at runtime; per-call overhead is negligible vs n.

using dot_fn = float (*)(const bf16_t*, const bf16_t*, int);

float dot_scalar(const bf16_t* w, const bf16_t* x, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return acc;
}

constexpr int PF_AHEAD = 512;

#if CPU_MOVA_X86
__attribute__((target("avx512f")))
float dot_avx512f(const bf16_t* w, const bf16_t* x, int n) {
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 64 <= n; i += 64) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 64; j += 16) {
      __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i + j));
      __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + j));
      __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
      __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
      __m512& acc = (j == 0) ? a0 : (j == 16) ? a1 : (j == 32) ? a2 : a3;
      acc = _mm512_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 16 <= n; i += 16) {
    __m256i wi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
    __m256i xi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    __m512 wf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(wi), 16));
    __m512 xf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(xi), 16));
    a0 = _mm512_fmadd_ps(wf, xf, a0);
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}

#if (defined(__GNUC__) && __GNUC__ >= 10) || defined(__clang__)
#define CPU_MOVA_HAS_AVX512BF16 1
__attribute__((target("avx512bf16,avx512f")))
static inline __m512bh load_bh(const bf16_t* p) {
  __m512i raw = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
  __m512bh out;
  std::memcpy(&out, &raw, sizeof(out));
  return out;
}

__attribute__((target("avx512bf16,avx512f")))
float dot_avx512bf16(const bf16_t* w, const bf16_t* x, int n) {
  __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
  __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
  int i = 0;
  for (; i + 128 <= n; i += 128) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
    a1 = _mm512_dpbf16_ps(a1, load_bh(w + i + 32), load_bh(x + i + 32));
    a2 = _mm512_dpbf16_ps(a2, load_bh(w + i + 64), load_bh(x + i + 64));
    a3 = _mm512_dpbf16_ps(a3, load_bh(w + i + 96), load_bh(x + i + 96));
  }
  for (; i + 32 <= n; i += 32) {
    a0 = _mm512_dpbf16_ps(a0, load_bh(w + i), load_bh(x + i));
  }
  float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // avx512bf16 available

__attribute__((target("avx2,fma")))
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  lo = _mm_add_ps(lo, _mm256_extractf128_ps(v, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
  return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma")))
float dot_avx2(const bf16_t* w, const bf16_t* x, int n) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    _mm_prefetch(reinterpret_cast<const char*>(w + i) + PF_AHEAD, _MM_HINT_T0);
    for (int j = 0; j < 32; j += 8) {
      __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i + j));
      __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i + j));
      __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
      __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
      __m256& acc = (j == 0) ? a0 : (j == 8) ? a1 : (j == 16) ? a2 : a3;
      acc = _mm256_fmadd_ps(wf, xf, acc);
    }
  }
  for (; i + 8 <= n; i += 8) {
    __m128i wi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
    __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));
    __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(wi), 16));
    __m256 xf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
    a0 = _mm256_fmadd_ps(wf, xf, a0);
  }
  float s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  for (; i < n; ++i) s += bf16_to_f32(w[i]) * bf16_to_f32(x[i]);
  return s;
}
#endif  // CPU_MOVA_X86

struct DotChoice {
  dot_fn fn;
  const char* name;
};

enum IsaTier { ISA_SCALAR = 0, ISA_AVX2 = 1, ISA_AVX512 = 2, ISA_AVX512BF16 = 3 };

inline IsaTier pick_isa() {
#if CPU_MOVA_X86
  if (getenv("FREETOKEN_CPU_MOVA_SCALAR")) return ISA_SCALAR;
  IsaTier best = ISA_SCALAR;
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) best = ISA_AVX2;
  if (best >= ISA_AVX2 && __builtin_cpu_supports("avx512f")) best = ISA_AVX512;
#ifdef CPU_MOVA_HAS_AVX512BF16
  if (best >= ISA_AVX512 && __builtin_cpu_supports("avx512bf16")) best = ISA_AVX512BF16;
#endif
  if (const char* f = getenv("FREETOKEN_CPU_MOVA_ISA")) {
    IsaTier want = best;
    if (!std::strcmp(f, "scalar")) want = ISA_SCALAR;
    else if (!std::strcmp(f, "avx2")) want = ISA_AVX2;
    else if (!std::strcmp(f, "avx512")) want = ISA_AVX512;
    else if (!std::strcmp(f, "avx512bf16")) want = ISA_AVX512BF16;
    if (want < best) best = want;
  }
  return best;
#else
  return ISA_SCALAR;
#endif
}

DotChoice select_dot() {
  const IsaTier t = pick_isa();
#if CPU_MOVA_X86
#ifdef CPU_MOVA_HAS_AVX512BF16
  if (t >= ISA_AVX512BF16) return {dot_avx512bf16, "avx512bf16"};
#endif
  if (t >= ISA_AVX512) return {dot_avx512f, "avx512f"};
  if (t >= ISA_AVX2) return {dot_avx2, "avx2"};
#endif
  (void)t;
  return {dot_scalar, "scalar"};
}

// Output-column block for the run split. K=1024 value dim / 32 = 32 blocks per
// token; decode (bs=1) fan-outs over all workers.
constexpr int COL_BLK = 32;

struct MovaTask {
  struct CpuMovaExecutor* exec;
  int layer_id;
  int num_tokens;
  const bf16_t* x;    // [tokens, H] pinned bf16
  const int32_t* ids; // [tokens, top_k] pinned int32 (expert index; -1 = skip)
  const float* w;     // [tokens, top_k] pinned fp32 (router weights)
  bf16_t* y;          // [tokens, K] pinned bf16
};

struct CpuMovaExecutor {
  int num_threads;
  int num_layers;   // absolute layers covered (pointer table is layer-major)
  int num_experts;  // E per layer
  int top_k;        // routes per token
  int H;            // hidden size (GEMV length)
  int K;            // value dim (GEMV outputs per expert)
  int max_tokens;
  const uint64_t* experts_tbl;  // [num_layers * E] -> [K, H] bf16 weight base
  dot_fn dot;
  std::string isa_str;
  const char* isa;

  std::vector<std::thread> workers;
  std::mutex task_mtx;
  std::condition_variable task_cv;
  std::mutex sync_mtx;
  std::condition_variable sync_cv;

  bool stop = false;
  uint64_t cur_gen = 0;
  MovaTask* cur_task = nullptr;
  std::atomic<uint64_t> submitted{0};
  std::atomic<uint64_t> completed{0};
  std::atomic<int64_t> next_unit{0};
  int64_t total_units = 0;
  std::atomic<int> done_count{0};
  std::vector<MovaTask*> owned_tasks;
  std::vector<int> core_ids;

  CpuMovaExecutor(int num_threads_, int num_layers_, int num_experts_, int top_k_,
                  int hidden_size, int value_dim, int max_tokens,
                  uintptr_t experts_ptr, std::vector<int> core_ids_)
      : num_threads(num_threads_ > 0 ? num_threads_ : 1),
        num_layers(num_layers_),
        num_experts(num_experts_),
        top_k(top_k_),
        H(hidden_size),
        K(value_dim),
        max_tokens(max_tokens),
        experts_tbl(reinterpret_cast<const uint64_t*>(experts_ptr)),
        core_ids(std::move(core_ids_)) {
    DotChoice c = select_dot();
    dot = c.fn;
    isa_str = std::string(c.name);
    isa = isa_str.c_str();
    for (int i = 0; i < num_threads; ++i)
      workers.emplace_back(&CpuMovaExecutor::worker_loop, this, i);
  }

  ~CpuMovaExecutor() {
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      stop = true;
    }
    task_cv.notify_all();
    for (auto& th : workers)
      if (th.joinable()) th.join();
    for (MovaTask* t : owned_tasks) delete t;
  }

  uintptr_t create_task(int layer_id, int num_tokens, uintptr_t x_ptr,
                        uintptr_t ids_ptr, uintptr_t w_ptr, uintptr_t y_ptr) {
    MovaTask* t = new MovaTask{this,
                               layer_id,
                               num_tokens,
                               reinterpret_cast<const bf16_t*>(x_ptr),
                               reinterpret_cast<const int32_t*>(ids_ptr),
                               reinterpret_cast<const float*>(w_ptr),
                               reinterpret_cast<bf16_t*>(y_ptr)};
    owned_tasks.push_back(t);
    return reinterpret_cast<uintptr_t>(t);
  }

  const char* isa_name() const { return isa; }

  void pin_self(int tid) {
#if CPU_MOVA_HAS_AFFINITY
    if (core_ids.empty()) return;
    const int cpu = core_ids[tid % static_cast<int>(core_ids.size())];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)tid;
#endif
  }

  // One (token, col-block) unit: y[tok][c0:c1] = sum_k silu(dot(W[e_k][c], x)) * w_k.
  void do_unit(const MovaTask* t, int64_t p, int64_t n_col_blk) {
    const int64_t cb = p % n_col_blk;
    const int tok = static_cast<int>(p / n_col_blk);
    const int c0 = static_cast<int>(cb) * COL_BLK;
    const int c1 = std::min(K, c0 + COL_BLK);
    const bf16_t* x_row = t->x + (size_t)tok * H;
    bf16_t* y_row = t->y + (size_t)tok * K;
    for (int c = c0; c < c1; ++c) {
      float acc = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const int e = t->ids[(size_t)tok * top_k + k];
        if (e < 0 || e >= num_experts) continue;
        const float wv = t->w[(size_t)tok * top_k + k];
        const bf16_t* W =
            reinterpret_cast<const bf16_t*>(experts_tbl[(size_t)t->layer_id * num_experts + e]);
        const float v = dot(W + (size_t)c * H, x_row, H);
        acc += silu_f(v) * wv;
      }
      y_row[c] = f32_to_bf16(acc);
    }
  }

  void run_task_body(const MovaTask* t) {
    const int64_t n_col_blk = (K + COL_BLK - 1) / COL_BLK;
    for (;;) {
      int64_t p = next_unit.fetch_add(1, std::memory_order_relaxed);
      if (p >= total_units) break;
      do_unit(t, p, n_col_blk);
    }
  }

  void worker_loop(int tid) {
    pin_self(tid);
    uint64_t my_gen = 0;
    for (;;) {
      MovaTask* t;
      {
        std::unique_lock<std::mutex> lk(task_mtx);
        task_cv.wait(lk, [&] { return stop || cur_gen != my_gen; });
        if (stop) return;
        my_gen = cur_gen;
        t = cur_task;
      }
      run_task_body(t);
      if (done_count.fetch_add(1) + 1 == num_threads) {
        completed.store(my_gen, std::memory_order_release);
        {
          std::lock_guard<std::mutex> lk(sync_mtx);
        }
        sync_cv.notify_all();
      }
    }
  }

  void submit(MovaTask* t) {
    const int64_t n_col_blk = (K + COL_BLK - 1) / COL_BLK;
    total_units = (int64_t)t->num_tokens * n_col_blk;
    next_unit.store(0, std::memory_order_relaxed);
    done_count.store(0, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(task_mtx);
      cur_task = t;
      ++cur_gen;
      submitted.store(cur_gen, std::memory_order_release);
    }
    task_cv.notify_all();
  }

  void sync() {
    const uint64_t target = submitted.load(std::memory_order_acquire);
    std::unique_lock<std::mutex> lk(sync_mtx);
    sync_cv.wait(lk, [&] { return completed.load(std::memory_order_acquire) >= target; });
  }

  void submit_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMovaExecutor::submit_cb,
                       reinterpret_cast<void*>(task));
  }

  void sync_with_cuda_stream(uintptr_t stream, uintptr_t task) {
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &CpuMovaExecutor::sync_cb,
                       reinterpret_cast<void*>(task));
  }

  // Eager (non-graph) path: run one task to completion on the pool.
  void run_task(uintptr_t task) {
    MovaTask* t = reinterpret_cast<MovaTask*>(task);
    submit(t);
    sync();
  }

  static void CUDART_CB submit_cb(void* ud) {
    MovaTask* t = reinterpret_cast<MovaTask*>(ud);
    t->exec->submit(t);
  }
  static void CUDART_CB sync_cb(void* ud) {
    MovaTask* t = reinterpret_cast<MovaTask*>(ud);
    t->exec->sync();
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  py::class_<CpuMovaExecutor>(m, "CpuMovaExecutor")
      .def(py::init<int, int, int, int, int, int, int, uintptr_t, std::vector<int>>(),
           py::arg("num_threads"), py::arg("num_layers"), py::arg("num_experts"),
           py::arg("top_k"), py::arg("hidden_size"), py::arg("value_dim"),
           py::arg("max_tokens"), py::arg("experts_ptr"), py::arg("core_ids"))
      .def("create_task", &CpuMovaExecutor::create_task, py::arg("layer_id"),
           py::arg("num_tokens"), py::arg("x_ptr"), py::arg("ids_ptr"), py::arg("w_ptr"),
           py::arg("y_ptr"))
      .def("submit_with_cuda_stream", &CpuMovaExecutor::submit_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("sync_with_cuda_stream", &CpuMovaExecutor::sync_with_cuda_stream,
           py::arg("stream"), py::arg("task"), py::call_guard<py::gil_scoped_release>())
      .def("run_task", &CpuMovaExecutor::run_task, py::arg("task"),
           py::call_guard<py::gil_scoped_release>())
      .def("isa_name", &CpuMovaExecutor::isa_name);
}
