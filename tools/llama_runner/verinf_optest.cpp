// One operation on identical inputs through a chosen backend's kernels
// (analysis/k2-fidelity-study-design.md §7.1, review of 2026-10-21): the
// prefill discrepancy is located first by prover/prefill_localize.py; the
// operation it finds is then run here alone, all its tokens in one graph (the
// batched kernels) or one token per graph (the one-token kernels), on the GPU
// and on the CPU, so the kernels' arithmetic is compared on the same inputs.
//
//   verinf_optest --op mul_mat|mul_mat_id --device CUDA0|CPU --split all|single --out DIR
//       --a A.f32 --a-ne ne0,ne1,ne2,ne3 --a-type f32|f16|bf16     src0 from raw float32, cast as given
//     | --a-gguf MODEL.gguf --a-tensor NAME                          src0 from the GGUF, its own type
//       --b B.f32 --b-ne ne0,ne1,ne2,ne3 [--b-permute 0,2,1,3]      src1, float32, permuted in the graph
//       [--ids IDS.i32 --ids-ne n_used,n_tokens]                     mul_mat_id's expert ids
//       [--token-dim 1|2]  the token dimension of src1 as the operation sees it (after any permute)
//       [--prec f32]       ggml_prec_set_acc(…, GGML_PREC_F32), as llama.cpp's attention scores
//
// Output in DIR: out.f32, the result of each graph in turn (one graph: all tokens;
// single: one per token, each chunk in ggml order, chunk_ne in meta.json), and
// meta.json (the chunks' shape and count, the backend, src0's type, timing).
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string & m) {
    fprintf(stderr, "verinf_optest: %s\n", m.c_str());
    exit(2);
}

std::vector<int64_t> ints(const std::string & s) {
    std::vector<int64_t> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) v.push_back(std::stoll(x));
    return v;
}

std::array<int64_t, 4> ne4(const std::string & s) {
    auto v = ints(s);
    if (v.empty() || v.size() > 4) die("a shape is 1 to 4 sizes: " + s);
    std::array<int64_t, 4> ne = {1, 1, 1, 1};
    for (size_t i = 0; i < v.size(); i++) ne[i] = v[i];
    return ne;
}

template <typename T> std::vector<T> read_raw(const std::string & path, size_t n) {
    std::vector<T> v(n);
    FILE * f = fopen(path.c_str(), "rb");
    if (!f || fread(v.data(), sizeof(T), n, f) != n) die("cannot read " + std::to_string(n) + " values from " + path);
    fclose(f);
    return v;
}

struct Args {
    std::string op = "mul_mat", device = "CUDA0", split = "all", out;
    std::string a, a_type = "f32", a_gguf, a_tensor, b, ids;
    std::array<int64_t, 4> a_ne{}, b_ne{};
    std::vector<int64_t> b_permute, ids_ne;
    int token_dim = 1;
    bool prec_f32 = false;
};

Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto v = [&]() -> std::string { if (i + 1 >= argc) die("missing value for " + k); return argv[++i]; };
        if (k == "--op") a.op = v();
        else if (k == "--device") a.device = v();
        else if (k == "--split") a.split = v();
        else if (k == "--out") a.out = v();
        else if (k == "--a") a.a = v();
        else if (k == "--a-ne") a.a_ne = ne4(v());
        else if (k == "--a-type") a.a_type = v();
        else if (k == "--a-gguf") a.a_gguf = v();
        else if (k == "--a-tensor") a.a_tensor = v();
        else if (k == "--b") a.b = v();
        else if (k == "--b-ne") a.b_ne = ne4(v());
        else if (k == "--b-permute") a.b_permute = ints(v());
        else if (k == "--ids") a.ids = v();
        else if (k == "--ids-ne") a.ids_ne = ints(v());
        else if (k == "--token-dim") a.token_dim = std::stoi(v());
        else if (k == "--prec") a.prec_f32 = v() == "f32";
        else die("unknown argument " + k);
    }
    if (a.out.empty() || a.b.empty()) die("--out and --b are required");
    if (a.a.empty() == a.a_gguf.empty()) die("src0: --a or --a-gguf, one of them");
    if (a.op != "mul_mat" && a.op != "mul_mat_id") die("op: mul_mat|mul_mat_id");
    if (a.op == "mul_mat_id" && (a.ids.empty() || a.ids_ne.size() != 2)) die("mul_mat_id needs --ids and --ids-ne");
    if (a.split != "all" && a.split != "single") die("split: all|single");
    if (!a.b_permute.empty() && a.b_permute.size() != 4) die("--b-permute names 4 axes");
    return a;
}

ggml_type type_of(const std::string & s) {
    if (s == "f32") return GGML_TYPE_F32;
    if (s == "f16") return GGML_TYPE_F16;
    if (s == "bf16") return GGML_TYPE_BF16;
    die("a-type " + s);
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char ** argv) {
    Args a = parse(argc, argv);
    ggml_backend_load_all();
    ggml_backend_t be = a.device == "CPU" ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)
                                          : ggml_backend_init_by_name(a.device.c_str(), nullptr);
    if (!be) die("no backend " + a.device);

    // src0, allocated once on the backend and shared by every graph
    ggml_init_params pa = {ggml_tensor_overhead() * 4, nullptr, true};
    ggml_context * ctx_a = ggml_init(pa);
    ggml_tensor * A;
    std::string a_type;
    std::vector<uint8_t> a_bytes;
    if (!a.a_gguf.empty()) {
        ggml_context * meta = nullptr;
        gguf_init_params gp = {true, &meta};
        gguf_context * g = gguf_init_from_file(a.a_gguf.c_str(), gp);
        if (!g) die("cannot read " + a.a_gguf);
        const int64_t id = gguf_find_tensor(g, a.a_tensor.c_str());
        if (id < 0) die("no tensor " + a.a_tensor);
        ggml_tensor * src = ggml_get_tensor(meta, a.a_tensor.c_str());
        A = ggml_new_tensor(ctx_a, src->type, ggml_n_dims(src), src->ne);
        a_type = ggml_type_name(src->type);
        a_bytes.resize(ggml_nbytes(src));
        FILE * f = fopen(a.a_gguf.c_str(), "rb");
        if (!f || fseeko(f, (off_t) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id)), SEEK_SET) != 0 ||
            fread(a_bytes.data(), 1, a_bytes.size(), f) != a_bytes.size()) die("cannot read the tensor's data");
        fclose(f);
        gguf_free(g);
        ggml_free(meta);
    } else {
        const ggml_type t = type_of(a.a_type);
        A = ggml_new_tensor_4d(ctx_a, t, a.a_ne[0], a.a_ne[1], a.a_ne[2], a.a_ne[3]);
        a_type = a.a_type;
        auto f32 = read_raw<float>(a.a, (size_t) ggml_nelements(A));
        a_bytes.resize(ggml_nbytes(A));
        if (t == GGML_TYPE_F32) memcpy(a_bytes.data(), f32.data(), a_bytes.size());
        else if (t == GGML_TYPE_F16) ggml_fp32_to_fp16_row(f32.data(), (ggml_fp16_t *) a_bytes.data(), (int64_t) f32.size());
        else ggml_fp32_to_bf16_row(f32.data(), (ggml_bf16_t *) a_bytes.data(), (int64_t) f32.size());
    }
    ggml_backend_buffer_t buf_a = ggml_backend_alloc_ctx_tensors(ctx_a, be);
    ggml_backend_tensor_set(A, a_bytes.data(), 0, a_bytes.size());

    const auto & bn = a.b_ne;
    auto b_all = read_raw<float>(a.b, (size_t) (bn[0] * bn[1] * bn[2] * bn[3]));
    std::vector<int32_t> ids_all;
    if (a.op == "mul_mat_id") ids_all = read_raw<int32_t>(a.ids, (size_t) (a.ids_ne[0] * a.ids_ne[1]));

    // the token dimension of src1 as stored: before the permute, if any
    int tok_stored = a.token_dim;
    if (!a.b_permute.empty())
        for (int d = 0; d < 4; d++) if (a.b_permute[d] == a.token_dim) tok_stored = d;
    const int64_t T = bn[tok_stored];
    if (a.op == "mul_mat_id" && a.ids_ne[1] != T) die("ids' tokens differ from src1's");
    const int64_t per = a.split == "all" ? T : 1;

    std::vector<float> out;
    std::array<int64_t, 4> chunk_ne{};
    int64_t chunks = 0;
    double t_compute = 0;
    for (int64_t t0 = 0; t0 < T; t0 += per) {
        // src1's tokens [t0, t0 + per), contiguous in its stored layout
        std::array<int64_t, 4> ne = bn;
        ne[tok_stored] = per;
        std::vector<float> bt((size_t) (ne[0] * ne[1] * ne[2] * ne[3]));
        size_t k = 0;
        for (int64_t i3 = 0; i3 < ne[3]; i3++)
        for (int64_t i2 = 0; i2 < ne[2]; i2++)
        for (int64_t i1 = 0; i1 < ne[1]; i1++)
        for (int64_t i0 = 0; i0 < ne[0]; i0++) {
            std::array<int64_t, 4> s = {i0, i1, i2, i3};
            s[tok_stored] += t0;
            bt[k++] = b_all[(size_t) (((s[3] * bn[2] + s[2]) * bn[1] + s[1]) * bn[0] + s[0])];
        }
        ggml_init_params pb = {ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
        ggml_context * ctx = ggml_init(pb);
        ggml_tensor * B = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
        ggml_tensor * Bv = a.b_permute.empty() ? B
            : ggml_permute(ctx, B, (int) a.b_permute[0], (int) a.b_permute[1], (int) a.b_permute[2], (int) a.b_permute[3]);
        ggml_tensor * ids = nullptr;
        ggml_tensor * C;
        if (a.op == "mul_mat_id") {
            ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, a.ids_ne[0], per);
            C = ggml_mul_mat_id(ctx, A, Bv, ids);
        } else {
            C = ggml_mul_mat(ctx, A, Bv);
        }
        if (a.prec_f32) ggml_prec_set_acc(C, GGML_PREC_F32);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(B, bt.data(), 0, ggml_nbytes(B));
        if (ids) ggml_backend_tensor_set(ids, ids_all.data() + (size_t) (t0 * a.ids_ne[0]), 0, ggml_nbytes(ids));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, C);
        const double c0 = now();
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) die("compute failed");
        t_compute += now() - c0;
        std::vector<float> ct((size_t) ggml_nelements(C));
        ggml_backend_tensor_get(C, ct.data(), 0, ggml_nbytes(C));
        out.insert(out.end(), ct.begin(), ct.end());
        for (int d = 0; d < 4; d++) chunk_ne[d] = C->ne[d];
        chunks++;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    std::string mk = "mkdir -p '" + a.out + "'";
    if (system(mk.c_str()) != 0) die("cannot create " + a.out);
    FILE * f = fopen((a.out + "/out.f32").c_str(), "wb");
    if (!f || fwrite(out.data(), sizeof(float), out.size(), f) != out.size()) die("write out.f32");
    fclose(f);
    std::ofstream(a.out + "/meta.json") << "{\"op\": \"" << a.op << "\", \"device\": \"" << a.device
        << "\", \"backend\": \"" << ggml_backend_name(be) << "\", \"split\": \"" << a.split
        << "\", \"src0_type\": \"" << a_type << "\", \"prec\": \"" << (a.prec_f32 ? "f32" : "default")
        << "\", \"tokens\": " << T << ", \"chunks\": " << chunks << ", \"chunk_ne\": [" << chunk_ne[0] << ", "
        << chunk_ne[1] << ", " << chunk_ne[2] << ", " << chunk_ne[3] << "], \"compute_s\": " << t_compute << "}\n";
    ggml_backend_buffer_free(buf_a);
    ggml_free(ctx_a);
    ggml_backend_free(be);
    return 0;
}
