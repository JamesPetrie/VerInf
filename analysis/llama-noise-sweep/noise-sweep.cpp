// noise-sweep: measure noise amplification through Llama-4-Maverick's native
// llama.cpp inference kernels.
//
// Phase 1 (baseline): decode 14 known tokens, capture clean input embeddings
// (graph node "embd", the get_rows output; named "inp_embd" in older versions),
// per-MoE-layer expert routing ("ffn_moe_topk-<il>") and full logits at all
// positions. Validates the known greedy continuation.
//
// Phase 2 (sweep): for each noise level (frac, step), perturb a `frac`
// fraction of the clean n_embd x 14 embedding entries by +/- step*(1/4096)
// (mt19937 seed 11, re-seeded per level), clear the KV cache, decode a batch
// that supplies EMBEDDINGS (llama_batch_init with embd = n_embd), and report
// routing flips, greedy agreement vs baseline argmax, and unexplained-info
// estimates U_sm and U_ln(sigma).
//
// Modeled on examples/eval-callback (cb_eval via llama_context_params).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

static const char * k_default_model =
    "/home/amodo/maverick-gguf/UD-Q4_K_XL/Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-00001-of-00005.gguf";

// the 14-token sequence: 2-token prompt + 12-token known greedy continuation
static const std::vector<llama_token> k_tokens = {
    200000, 954, 2182, 373, 262, 17252, 323, 1092, 954, 6076, 323, 12311, 25, 656
};
// known greedy continuation (argmax at positions 1..12 must produce tokens 2..13)
static const std::vector<llama_token> k_continuation = {
    2182, 373, 262, 17252, 323, 1092, 954, 6076, 323, 12311, 25, 656
};

struct cb_data {
    bool capture_embd = false;   // only needed during phase 1
    int  n_tokens     = 0;
    int  n_embd       = 0;

    bool        embd_captured = false;
    std::string embd_node_name;
    std::vector<float> embd;     // n_embd * n_tokens, token-major

    // layer index -> selected expert ids, k * n_tokens (token-major, k = n_expert_used)
    std::map<int, std::vector<int32_t>> routing;

    // layer index -> raw router logits, n_expert * n_tokens (token-major)
    std::map<int, std::vector<float>> router_logits;
};

static bool cb_eval_fn(struct ggml_tensor * t, bool ask, void * user_data) {
    cb_data * d = (cb_data *) user_data;
    const char * name = t->name;

    const bool is_topk   = strncmp(name, "ffn_moe_topk-",   13) == 0 && t->type == GGML_TYPE_I32;
    const bool is_rlogit = strncmp(name, "ffn_moe_logits-", 15) == 0 && t->type == GGML_TYPE_F32;
    const bool is_embd = d->capture_embd && !d->embd_captured &&
                         (strcmp(name, "embd") == 0 || strcmp(name, "inp_embd") == 0) &&
                         t->type == GGML_TYPE_F32 &&
                         (int) t->ne[0] >= d->n_embd && (int) t->ne[1] == d->n_tokens;

    if (ask) {
        return is_topk || is_embd || is_rlogit;
    }

    if (is_embd) {
        d->embd.resize((size_t) d->n_embd * d->n_tokens);
        for (int i = 0; i < d->n_tokens; ++i) {
            // copy row-by-row in case the tensor is a strided view
            ggml_backend_tensor_get(t, d->embd.data() + (size_t) i * d->n_embd,
                                    (size_t) i * t->nb[1], (size_t) d->n_embd * sizeof(float));
        }
        d->embd_captured  = true;
        d->embd_node_name = name;
    }

    if (is_rlogit) {
        const int il = atoi(name + 15);
        const int ne = (int) t->ne[0];   // n_expert
        const int nt = (int) t->ne[1];
        auto & v = d->router_logits[il];
        v.resize((size_t) ne * nt);
        for (int i = 0; i < nt; ++i) {
            ggml_backend_tensor_get(t, v.data() + (size_t) i * ne,
                                    (size_t) i * t->nb[1], (size_t) ne * sizeof(float));
        }
    }

    if (is_topk) {
        const int il = atoi(name + 13);
        const int k  = (int) t->ne[0];   // n_expert_used (1 for Maverick)
        const int nt = (int) t->ne[1];
        auto & v = d->routing[il];
        v.resize((size_t) k * nt);
        for (int i = 0; i < nt; ++i) {
            ggml_backend_tensor_get(t, v.data() + (size_t) i * k,
                                    (size_t) i * t->nb[1], (size_t) k * sizeof(int32_t));
        }
    }

    return true;
}

// -log2 softmax(l)[o], max-subtracted log-sum-exp
static double u_softmax(const float * l, int n, int o) {
    float mx = l[0];
    for (int j = 1; j < n; ++j) mx = std::max(mx, l[j]);
    double sum = 0.0;
    for (int j = 0; j < n; ++j) sum += std::exp((double) l[j] - (double) mx);
    return (std::log(sum) + (double) mx - (double) l[o]) / std::log(2.0);
}

// logit-noise estimator: w_j = exp(-(max(l)-l_j)^2/(2 sigma^2)); U = -log2(w_o / sum_j w_j)
// computed as d_o^2/(2 sigma^2 ln2) + log2(sum) to avoid w_o underflow
static double u_logit_noise(const float * l, int n, int o, double sigma) {
    float mx = l[0];
    for (int j = 1; j < n; ++j) mx = std::max(mx, l[j]);
    const double inv2s2 = 1.0 / (2.0 * sigma * sigma);
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dd = (double) mx - (double) l[j];
        sum += std::exp(-dd * dd * inv2s2);
    }
    const double do_ = (double) mx - (double) l[o];
    return (do_ * do_ * inv2s2 + std::log(sum)) / std::log(2.0);
}

static int argmax_f(const float * l, int n) {
    int best = 0;
    for (int j = 1; j < n; ++j) if (l[j] > l[best]) best = j;
    return best;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    cb_data data;

    common_params params;
    params.model.path = k_default_model;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = cb_eval_fn;
    params.cb_eval_user_data = &data;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const int n_vocab          = llama_vocab_n_tokens(vocab);
    const int n_embd           = llama_model_n_embd(model);
    const int n_embd_inp       = llama_model_n_embd_inp(model);
    const int N                = (int) k_tokens.size();

    LOG_INF("noise-sweep: n_vocab = %d, n_embd = %d, n_embd_inp = %d, n_tokens = %d\n",
            n_vocab, n_embd, n_embd_inp, N);
    if (n_embd_inp != n_embd) {
        LOG_ERR("noise-sweep: n_embd_inp (%d) != n_embd (%d) - embd batch path untested, aborting\n",
                n_embd_inp, n_embd);
        return 1;
    }

    data.n_tokens = N;
    data.n_embd   = n_embd;

    // ---------------- Phase 1: token-input baseline ----------------
    const auto t0 = std::chrono::steady_clock::now();
    {
        llama_batch batch = llama_batch_init(N, 0, 1);
        batch.n_tokens = N;
        for (int i = 0; i < N; ++i) {
            batch.token[i]     = k_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        data.capture_embd = true;
        data.routing.clear();
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("noise-sweep: baseline llama_decode failed\n");
            return 1;
        }
        llama_batch_free(batch);
        data.capture_embd = false;
    }
    const double t_base = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (!data.embd_captured) {
        LOG_ERR("noise-sweep: failed to capture clean input embeddings (no 'embd'/'inp_embd' node fired)\n");
        return 1;
    }
    LOG_INF("noise-sweep: captured clean embeddings from node '%s'\n", data.embd_node_name.c_str());
    LOG_INF("noise-sweep: captured routing for %zu MoE layers\n", data.routing.size());

    const std::vector<float> clean_embd = data.embd;
    const std::map<int, std::vector<int32_t>> base_routing = data.routing;
    const std::map<int, std::vector<float>>   base_rlogits = data.router_logits;

    int n_route_entries = 0;
    for (const auto & kv : base_routing) n_route_entries += (int) kv.second.size();

    // baseline logits + argmax
    std::vector<float> base_logits((size_t) N * n_vocab);
    std::vector<int>   base_argmax(N);
    for (int i = 0; i < N; ++i) {
        const float * l = llama_get_logits_ith(ctx, i);
        memcpy(base_logits.data() + (size_t) i * n_vocab, l, (size_t) n_vocab * sizeof(float));
        base_argmax[i] = argmax_f(l, n_vocab);
    }

    // gate (b): baseline must reproduce the known greedy continuation
    bool gate_b = true;
    printf("\n[gate b] baseline argmax continuation (positions 1..13):\n");
    for (int p = 1; p < N; ++p) {
        const bool has_ref = (p - 1) < (int) k_continuation.size();
        const llama_token ref = has_ref ? k_continuation[p - 1] : -1;
        const bool ok = !has_ref || base_argmax[p] == ref;
        if (has_ref && !ok) gate_b = false;
        printf("  pos %2d: argmax = %6d  %s\n", p, base_argmax[p],
               has_ref ? (ok ? "(matches reference)" : "(MISMATCH vs reference!)") : "(next-token, no reference)");
    }
    printf("[gate b] %s\n", gate_b ? "PASS - known continuation reproduced" : "FAIL");
    fflush(stdout);

    // ---------------- Phase 2: embedding-input noise sweep ----------------
    struct level { double frac; int step; };
    const std::vector<level> levels = {
        {0.0,    0},   // zero-noise sanity row (gate c)
        {1e-4,   1},
        {1e-3,   1},
        {1e-2,   1},
        {1e-1,   1},
        {1.0,    1},
        {1.0,   16},
        {1.0,  256},
    };

    llama_batch ebatch = llama_batch_init(N, n_embd, 1);
    ebatch.n_tokens = N;
    for (int i = 0; i < N; ++i) {
        ebatch.pos[i]       = i;
        ebatch.n_seq_id[i]  = 1;
        ebatch.seq_id[i][0] = 0;
        ebatch.logits[i]    = 1;
    }

    const size_t n_entries = (size_t) n_embd * N;
    std::vector<std::string> table;
    table.push_back("frac      step  n_flip   router_flips  greedy_agree  U_sm        U_ln(0.3)   U_ln(1.0)   U_ln(3.0)   secs");

    bool gate_c = true, gate_d1 = true, gate_d2 = true;

    for (const auto & lv : levels) {
        const auto tl0 = std::chrono::steady_clock::now();

        // perturb
        std::vector<float> noisy = clean_embd;
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        const float delta = (float) lv.step * (1.0f / 4096.0f);
        long n_flip = 0;
        if (lv.frac > 0.0) {
            for (size_t e = 0; e < n_entries; ++e) {
                if (unif(rng) < lv.frac) {
                    noisy[e] += (rng() & 1) ? delta : -delta;
                    ++n_flip;
                }
            }
        }

        memcpy(ebatch.embd, noisy.data(), n_entries * sizeof(float));

        llama_memory_clear(llama_get_memory(ctx), true);
        data.routing.clear();
        data.router_logits.clear();

        if (llama_decode(ctx, ebatch) != 0) {
            LOG_ERR("noise-sweep: embd-input llama_decode failed at frac=%g step=%d\n", lv.frac, lv.step);
            return 1;
        }

        // routing flips vs baseline (with near-tie diagnostics from raw router logits)
        int router_flips = 0, route_total = 0;
        std::string flip_diag;
        for (const auto & kv : base_routing) {
            const auto it = data.routing.find(kv.first);
            if (it == data.routing.end() || it->second.size() != kv.second.size()) {
                LOG_ERR("noise-sweep: routing capture mismatch at layer %d\n", kv.first);
                return 1;
            }
            const int k = (int) (kv.second.size() / N);
            for (size_t j = 0; j < kv.second.size(); ++j) {
                ++route_total;
                if (it->second[j] != kv.second[j]) {
                    ++router_flips;
                    if (router_flips <= 24) {
                        const int tok = (int) j / k;
                        const int e_old = kv.second[j], e_new = it->second[j];
                        double margin = 0.0;
                        const auto bl = base_rlogits.find(kv.first);
                        if (bl != base_rlogits.end()) {
                            const int ne = (int) (bl->second.size() / N);
                            const float * rl = bl->second.data() + (size_t) tok * ne;
                            // baseline margin: top1 logit minus best other-expert logit
                            float best_other = -1e30f;
                            for (int x = 0; x < ne; ++x) if (x != e_old) best_other = std::max(best_other, rl[x]);
                            margin = (double) rl[e_old] - (double) best_other;
                        }
                        char buf[96];
                        snprintf(buf, sizeof(buf), "  L%d:t%d %d->%d (base margin %.4g)",
                                 kv.first, tok, e_old, e_new, margin);
                        flip_diag += buf;
                    }
                }
            }
        }

        // logits-based stats over positions 1..13
        int greedy_agree = 0;
        double u_sm = 0.0, u03 = 0.0, u10 = 0.0, u30 = 0.0;
        double max_logit_diff = 0.0;
        for (int p = 0; p < N; ++p) {
            const float * l = llama_get_logits_ith(ctx, p);
            if (lv.frac == 0.0) {
                const float * bl = base_logits.data() + (size_t) p * n_vocab;
                for (int j = 0; j < n_vocab; ++j) {
                    max_logit_diff = std::max(max_logit_diff, (double) std::fabs(l[j] - bl[j]));
                }
            }
            if (p == 0) continue;
            const int o = base_argmax[p];
            if (argmax_f(l, n_vocab) == o) ++greedy_agree;
            u_sm += u_softmax    (l, n_vocab, o);
            u03  += u_logit_noise(l, n_vocab, o, 0.3);
            u10  += u_logit_noise(l, n_vocab, o, 1.0);
            u30  += u_logit_noise(l, n_vocab, o, 3.0);
        }

        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tl0).count();

        char row[512];
        snprintf(row, sizeof(row),
                 "%-9.0e %4d  %6ld   %4d /%4d    %2d /13       %-11.4f %-11.4f %-11.4f %-11.4f %.1f",
                 lv.frac, lv.step, n_flip, router_flips, route_total, greedy_agree,
                 u_sm, u03, u10, u30, secs);
        table.push_back(row);
        printf("%s\n", row);
        if (!flip_diag.empty()) {
            printf("  flips:%s%s\n", flip_diag.c_str(), router_flips > 24 ? " ..." : "");
        }
        fflush(stdout);

        if (lv.frac == 0.0) {
            const bool same_argmax = greedy_agree == N - 1;
            gate_c = (router_flips == 0) && same_argmax;
            printf("[gate c] zero-noise embd-input run: router_flips = %d, greedy_agree = %d/13, "
                   "max |logit diff| vs token baseline = %.6g -> %s\n",
                   router_flips, greedy_agree, max_logit_diff, gate_c ? "PASS" : "FAIL");
            fflush(stdout);
        }
        if (lv.frac == 1e-4 && lv.step == 1) gate_d1 = (router_flips == 0);
        if (lv.frac == 1.0  && lv.step == 256) gate_d2 = (router_flips >= 20); // "substantial", numpy found 89/336
    }

    llama_batch_free(ebatch);

    // final summary table
    printf("\n================ noise sweep summary (Llama-4-Maverick, native llama.cpp kernels) ================\n");
    printf("baseline token-input decode: %.1f s; routing entries per run: %d; positions scored: 13\n",
           t_base, n_route_entries);
    printf("noise model: each selected entry of the clean (n_embd x 14) input embeddings perturbed by +/- step/4096, mt19937 seed 11\n");
    for (const auto & r : table) printf("%s\n", r.c_str());
    printf("\nvalidation gates: (b) baseline continuation %s; (c) zero-noise embd path %s; "
           "(d) 1e-4 row zero flips %s; (1.0,256) substantial flips %s\n",
           gate_b ? "PASS" : "FAIL", gate_c ? "PASS" : "FAIL",
           gate_d1 ? "PASS" : "FAIL", gate_d2 ? "PASS" : "FAIL");
    fflush(stdout);

    llama_backend_free();
    return (gate_b && gate_c) ? 0 : 2;
}
