// The engine arm of the fidelity study (analysis/k2-fidelity-study-design.md
// §3.3, §5.1): token ids in; logits, named layer tensors and a manifest out.
//
//   generate  the baseline: the prompt prefilled in chunks of --prompt-chunk,
//             then greedy one-token decoding (lowest id on a tie) for at most
//             --n-gen tokens or to EOS. The logits that chose each token are
//             recorded as they are produced; the last token is never decoded.
//   replay    a frozen sequence: the prompt in chunks of --prompt-chunk, the
//             continuation in chunks of --cont-chunk (512 and 512 is the
//             diagnostic; the prompt chunk and 1 reproduces generation).
//   info      load the model on the CPU, memory-mapped and not repacked, for the
//             loader-branch fingerprint (llama.cpp's print_info lines in load.log);
//             vocab_only skips those lines and no_alloc cannot load tensors.
//
// Devices: --split-mode none|layer|row, --main-gpu and --tensor-split (comma-
// separated proportions) as llama.cpp's; the settings, every device's memory
// after the context is made, and the whole log (run.log, written at exit, with
// the scheduler's placement when GGML_SCHED_DEBUG is set) are kept, so where
// each operation ran is recorded, not assumed.
//
// Outputs in --out: logits.f32 (row-major, one row per scored position, i.e.
// positions n_prompt-1 .. T-2), capture/<name>.f32 (positions x elements, all
// positions in order), tokens.txt (the full sequence), load.log, and
// manifest.json. Every position is an output when anything is captured, so
// the last layer's tensors cover every position.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef VERINF_LLAMA_COMMIT
#define VERINF_LLAMA_COMMIT "unknown"
#endif

namespace {

struct Args {
    std::string model, tokens, out, mode = "generate";
    int n_prompt = -1, n_gen = 0, prompt_chunk = 512, cont_chunk = 1, pos_offset = 0;
    int n_batch = 512, n_ubatch = 512, n_ctx = 0, threads = 0, n_gpu_layers = 0;
    std::string kv_type = "f16", flash_attn = "auto", split_mode = "layer";
    int main_gpu = 0;
    std::vector<float> tensor_split;
    float yarn_beta_fast = -1.0f, yarn_beta_slow = -1.0f;
    std::vector<std::string> capture;
    std::vector<std::string> overrides;   // --override-tensor pattern=buft (e.g. exps=CPU)
    bool extra_bufts = true;
    bool captures_optional = false;   // a capture never seen is listed, not an error
};

[[noreturn]] void die(const std::string & m) {
    fprintf(stderr, "verinf_runner: %s\n", m.c_str());
    exit(2);
}

std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, sep)) if (!x.empty()) out.push_back(x);
    return out;
}

Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto v = [&]() -> std::string { if (i + 1 >= argc) die("missing value for " + k); return argv[++i]; };
        if (k == "--model") a.model = v();
        else if (k == "--tokens") a.tokens = v();
        else if (k == "--out") a.out = v();
        else if (k == "--mode") a.mode = v();
        else if (k == "--n-prompt") a.n_prompt = std::stoi(v());
        else if (k == "--n-gen") a.n_gen = std::stoi(v());
        else if (k == "--prompt-chunk") a.prompt_chunk = std::stoi(v());
        else if (k == "--cont-chunk") a.cont_chunk = std::stoi(v());
        else if (k == "--pos-offset") a.pos_offset = std::stoi(v());
        else if (k == "--n-batch") a.n_batch = std::stoi(v());
        else if (k == "--n-ubatch") a.n_ubatch = std::stoi(v());
        else if (k == "--n-ctx") a.n_ctx = std::stoi(v());
        else if (k == "--threads") a.threads = std::stoi(v());
        else if (k == "--n-gpu-layers") a.n_gpu_layers = std::stoi(v());
        else if (k == "--kv-type") a.kv_type = v();
        else if (k == "--flash-attn") a.flash_attn = v();
        else if (k == "--yarn-beta-fast") a.yarn_beta_fast = std::stof(v());
        else if (k == "--yarn-beta-slow") a.yarn_beta_slow = std::stof(v());
        else if (k == "--capture") a.capture = split(v(), ',');
        else if (k == "--override-tensor") a.overrides.push_back(v());
        else if (k == "--no-extra-bufts") a.extra_bufts = false;
        else if (k == "--captures-optional") a.captures_optional = true;
        else if (k == "--split-mode") a.split_mode = v();
        else if (k == "--main-gpu") a.main_gpu = std::stoi(v());
        else if (k == "--tensor-split") for (auto & x : split(v(), ',')) a.tensor_split.push_back(std::stof(x));
        else die("unknown argument " + k);
    }
    if (a.model.empty() || a.out.empty()) die("--model and --out are required");
    if (a.mode != "info" && a.tokens.empty()) die("--tokens is required");
    if (a.mode != "generate" && a.mode != "replay" && a.mode != "info") die("mode: generate|replay|info");
    if (a.split_mode != "none" && a.split_mode != "layer" && a.split_mode != "row") die("split mode: none|layer|row");
    return a;
}

// ---- load log: kept whole, so the loader-branch fingerprint can be read ----

std::string g_log;
std::string g_out;      // the whole log goes to <out>/run.log at exit

void log_cb(ggml_log_level level, const char * text, void *) {
    g_log += text;
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

// ---- captures ---------------------------------------------------------------

struct Capture {
    std::set<std::string> names;
    std::map<std::string, std::vector<float>> data;
    std::map<std::string, int64_t> width;     // elements per row
    std::map<std::string, std::string> dtype;
    std::map<std::string, std::vector<std::array<int64_t, 4>>> shapes;   // each call's ne
    std::set<std::string> skipped;             // asked for, but their rows are not contiguous
    int64_t calls = 0;
};

// a capture is asked for by the tensor's name ("kq-0"), or, where one name labels
// several tensors, by its name and the operation that made it ("kv_cmpr-0/MUL"),
// and its row width too ("q-0/MUL_MAT/1536")
std::string capture_key(const Capture & c, const ggml_tensor * t) {
    const std::string name = ggml_get_name(t);
    const std::string by_op = name + "/" + ggml_op_name(t->op);
    const std::string by_width = by_op + "/" + std::to_string(t->ne[0]);
    if (c.names.count(by_width)) return by_width;
    if (c.names.count(by_op)) return by_op;
    if (c.names.count(name)) return name;
    return "";
}

bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * c = static_cast<Capture *>(ud);
    const std::string name = capture_key(*c, t);
    if (name.empty()) return ask ? false : true;
    if (ask) return true;
    // any rank, read row by row (i3, i2, i1 in order, each call's shape kept); a tensor whose
    // rows are not contiguous (a permuted view) is skipped and listed; the row width may change
    // between calls (the attention's KV extent grows), each call's shape says how
    if (t->nb[0] != ggml_type_size(t->type)) {
        c->skipped.insert(name);
        return true;
    }
    const int64_t w = t->ne[0], rows = ggml_nelements(t) / t->ne[0];
    if (!c->width.count(name) || c->width[name] != w) c->width[name] = c->width.count(name) ? -1 : w;
    c->shapes[name].push_back({t->ne[0], t->ne[1], t->ne[2], t->ne[3]});
    auto & d = c->data[name];
    const size_t at = d.size();
    d.resize(at + (size_t) (w * rows));
    const size_t row_bytes = (size_t) w * ggml_type_size(t->type);
    std::vector<uint8_t> raw(row_bytes);
    int64_t r = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < t->ne[1]; i1++, r++) {
        const size_t off = (size_t) (i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
        float * dst = d.data() + at + (size_t) (r * w);
        if (t->type == GGML_TYPE_F32) {
            c->dtype[name] = "f32";
            ggml_backend_tensor_get(t, dst, off, row_bytes);
            continue;
        }
        ggml_backend_tensor_get(t, raw.data(), off, row_bytes);
        if (t->type == GGML_TYPE_F16) {
            c->dtype[name] = "f16-as-f32";
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), dst, w);
        } else if (t->type == GGML_TYPE_BF16) {
            c->dtype[name] = "bf16-as-f32";
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), dst, w);
        } else if (t->type == GGML_TYPE_I32) {
            c->dtype[name] = "i32-as-f32";
            for (int64_t i = 0; i < w; i++) dst[i] = (float) ((const int32_t *) raw.data())[i];
        } else {
            die("capture of an unsupported type for " + name);
        }
    }
    c->calls++;
    return true;
}

int64_t rows_of(Capture & c, const std::string & name) {
    int64_t n = 0;
    for (auto & ne : c.shapes[name]) n += ne[1] * ne[2] * ne[3];
    return n;
}

ggml_type kv_type(const std::string & s) {
    if (s == "f16") return GGML_TYPE_F16;
    if (s == "f32") return GGML_TYPE_F32;
    if (s == "bf16") return GGML_TYPE_BF16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    die("kv type " + s);
}

std::vector<llama_token> read_tokens(const std::string & path) {
    std::ifstream f(path);
    if (!f) die("cannot read " + path);
    std::vector<llama_token> ids;
    long long x;
    while (f >> x) ids.push_back((llama_token) x);
    return ids;
}

std::string jstr(const std::string & s) {
    std::string o = "\"";
    for (char ch : s) {
        if (ch == '"' || ch == '\\') { o += '\\'; o += ch; }
        else if (ch == '\n') o += "\\n";
        else if ((unsigned char) ch < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", ch); o += b; }
        else o += ch;
    }
    return o + "\"";
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char ** argv) {
    Args a = parse(argc, argv);
    std::string mk = "mkdir -p '" + a.out + "/capture'";
    if (system(mk.c_str()) != 0) die("cannot create " + a.out);
    llama_log_set(log_cb, nullptr);
    g_out = a.out;
    atexit([] { std::ofstream lf(g_out + "/run.log"); lf << g_log; });
    llama_backend_init();

    // tensor buffer-type overrides, as llama.cpp's --override-tensor
    std::vector<llama_model_tensor_buft_override> ov;
    std::vector<std::string> ov_pat;
    for (auto & o : a.overrides) {
        auto p = o.find('=');
        if (p == std::string::npos) die("override pattern=buffer-type: " + o);
        ov_pat.push_back(o.substr(0, p));
        std::string bt = o.substr(p + 1);
        ggml_backend_buffer_type_t buft = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            auto * dev = ggml_backend_dev_get(i);
            auto * t = ggml_backend_dev_buffer_type(dev);
            if (t && bt == ggml_backend_buft_name(t)) buft = t;
        }
        if (!buft) die("unknown buffer type " + bt);
        ov.push_back({nullptr, buft});
    }
    for (size_t i = 0; i < ov.size(); i++) ov[i].pattern = ov_pat[i].c_str();
    if (!ov.empty()) ov.push_back({nullptr, nullptr});

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = a.n_gpu_layers;
    mp.split_mode = a.split_mode == "none" ? LLAMA_SPLIT_MODE_NONE
                  : a.split_mode == "row" ? LLAMA_SPLIT_MODE_ROW : LLAMA_SPLIT_MODE_LAYER;
    mp.main_gpu = a.main_gpu;
    std::vector<float> ts(llama_max_devices(), 0.0f);
    if (a.tensor_split.size() > ts.size()) die("--tensor-split names more devices than llama.cpp supports");
    std::copy(a.tensor_split.begin(), a.tensor_split.end(), ts.begin());
    if (!a.tensor_split.empty()) mp.tensor_split = ts.data();
    mp.use_extra_bufts = a.extra_bufts && a.mode != "info";
    if (a.mode == "info") mp.n_gpu_layers = 0;
    if (!ov.empty()) mp.tensor_buft_overrides = ov.data();
    const double t_load0 = now();
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    {
        std::ofstream lf(a.out + "/load.log");
        lf << g_log;
    }
    if (!model) die("model load failed (see load.log)");
    const double t_load = now() - t_load0;

    std::ostringstream man;
    man << "{\n  \"llama_cpp_commit\": " << jstr(VERINF_LLAMA_COMMIT)
        << ",\n  \"system_info\": " << jstr(llama_print_system_info())
        << ",\n  \"model\": " << jstr(a.model) << ",\n  \"mode\": " << jstr(a.mode);
    std::string devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev = ggml_backend_dev_get(i);
        devs += std::string(i ? ", " : "") + jstr(std::string(ggml_backend_dev_name(dev)) + ": " +
                                                  ggml_backend_dev_description(dev));
    }
    man << ",\n  \"devices\": [" << devs << "]";

    if (a.mode == "info") {
        man << ",\n  \"load_s\": " << t_load << "\n}\n";
        std::ofstream(a.out + "/manifest.json") << man.str();
        llama_model_free(model);
        return 0;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> seq = read_tokens(a.tokens);
    for (auto id : seq) if (id < 0 || id >= n_vocab) die("token id out of range");
    const int n_prompt = a.n_prompt < 0 ? (int) seq.size() : a.n_prompt;
    if (n_prompt < 1 || n_prompt > (int) seq.size()) die("--n-prompt outside the sequence");
    if (a.mode == "generate") seq.resize(n_prompt);
    const int T_max = a.mode == "generate" ? n_prompt + a.n_gen : (int) seq.size();

    Capture cap;
    for (auto & n : a.capture) cap.names.insert(n);
    const bool all_outputs = !cap.names.empty();

    auto cp = llama_context_default_params();
    // the context size is part of the configuration (§3.3): pinned, never derived from the
    // sequence, so capped generation and an early-EOS replay run with the same one
    if (a.n_ctx <= 0) die("--n-ctx is required");
    if (a.pos_offset + T_max > a.n_ctx) die("the sequence does not fit --n-ctx");
    cp.n_ctx = a.n_ctx;
    cp.n_batch = a.n_batch;
    cp.n_ubatch = a.n_ubatch;
    if (a.threads > 0) { cp.n_threads = a.threads; cp.n_threads_batch = a.threads; }
    cp.type_k = kv_type(a.kv_type);
    cp.type_v = kv_type(a.kv_type);
    cp.flash_attn_type = a.flash_attn == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : a.flash_attn == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                       : LLAMA_FLASH_ATTN_TYPE_AUTO;
    cp.yarn_beta_fast = a.yarn_beta_fast;
    cp.yarn_beta_slow = a.yarn_beta_slow;
    cp.no_perf = false;
    if (all_outputs) { cp.cb_eval = eval_cb; cp.cb_eval_user_data = &cap; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("context creation failed");
    std::string mem;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev = ggml_backend_dev_get(i);
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        mem += std::string(i ? ", " : "") + "{\"device\": " + jstr(ggml_backend_dev_name(dev)) +
               ", \"free\": " + std::to_string(free) + ", \"total\": " + std::to_string(total) + "}";
    }

    std::vector<float> logits;          // scored rows, row-major
    int n_scored = 0;
    auto keep_row = [&](int idx) {
        const float * row = llama_get_logits_ith(ctx, idx);
        if (!row) die("no logits for an output row");
        logits.insert(logits.end(), row, row + n_vocab);
        n_scored++;
    };
    // decode positions [p0, p1) of seq; outputs at every position when capturing, otherwise
    // only where a row is scored (positions n_prompt-1 .. T-2) or the next token is chosen
    auto run = [&](int p0, int p1, int T_scored_end) {
        llama_batch b = llama_batch_init(p1 - p0, 0, 1);
        for (int p = p0; p < p1; p++) {
            const int i = p - p0;
            b.token[i] = seq[p];
            b.pos[i] = a.pos_offset + p;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            const bool scored = p >= n_prompt - 1 && p < T_scored_end;
            b.logits[i] = (all_outputs || scored) ? 1 : 0;
        }
        b.n_tokens = p1 - p0;
        if (llama_decode(ctx, b) != 0) die("llama_decode failed");
        for (int p = p0; p < p1; p++) {
            if (p >= n_prompt - 1 && p < T_scored_end) keep_row(p - p0);
        }
        llama_batch_free(b);
    };

    const double t0 = now();
    double t_prefill = 0, t_decode = 0;
    std::vector<llama_token> generated;
    bool ended_by_eog = false;
    if (a.mode == "generate") {
        // prefill: only the last prompt position is scored before decoding starts
        for (int p0 = 0; p0 < n_prompt; p0 += a.prompt_chunk)
            run(p0, std::min(n_prompt, p0 + a.prompt_chunk), n_prompt);
        t_prefill = now() - t0;
        for (int step = 0; step < a.n_gen; step++) {
            const float * row = logits.data() + (size_t) (n_scored - 1) * n_vocab;
            llama_token best = 0;
            for (int v = 1; v < n_vocab; v++) if (row[v] > row[best]) best = v;
            seq.push_back(best);
            generated.push_back(best);
            ended_by_eog = llama_vocab_is_eog(vocab, best);
            if (ended_by_eog || step + 1 == a.n_gen) break;
            const int p = (int) seq.size() - 1;
            run(p, p + 1, p + 1);
        }
        t_decode = now() - t0 - t_prefill;
        // the row after the last token scores nothing: drop it if a capture forced it
        n_scored = (int) generated.size();
        logits.resize((size_t) n_scored * n_vocab);
    } else {
        const int T = (int) seq.size();
        for (int p0 = 0; p0 < n_prompt; p0 += a.prompt_chunk)
            run(p0, std::min(n_prompt, p0 + a.prompt_chunk), T - 1);
        t_prefill = now() - t0;
        for (int p0 = n_prompt; p0 < T; p0 += a.cont_chunk)
            run(p0, std::min(T, p0 + a.cont_chunk), T - 1);
        t_decode = now() - t0 - t_prefill;
    }
    const double t_total = now() - t0;

    {
        FILE * f = fopen((a.out + "/logits.f32").c_str(), "wb");
        if (!f || fwrite(logits.data(), sizeof(float), logits.size(), f) != logits.size()) die("write logits");
        fclose(f);
        std::ofstream tf(a.out + "/tokens.txt");
        for (auto id : seq) tf << id << "\n";
    }
    std::string caps;
    for (auto & [name, d] : cap.data) {
        std::string file = name;                       // an op-qualified name's "/" is "@" on disk
        std::replace(file.begin(), file.end(), '/', '@');
        const std::string fn = a.out + "/capture/" + file + ".f32";
        FILE * f = fopen(fn.c_str(), "wb");
        if (!f || fwrite(d.data(), sizeof(float), d.size(), f) != d.size()) die("write capture " + name);
        fclose(f);
        std::string shapes;
        for (auto & ne : cap.shapes[name])
            shapes += std::string(shapes.empty() ? "" : ", ") + "[" + std::to_string(ne[0]) + ", " +
                      std::to_string(ne[1]) + ", " + std::to_string(ne[2]) + ", " + std::to_string(ne[3]) + "]";
        caps += std::string(caps.empty() ? "" : ", ") + jstr(name) + ": {\"width\": " +
                std::to_string(cap.width[name]) + ", \"rows\": " +
                std::to_string((long long) rows_of(cap, name)) +
                ", \"dtype\": " + jstr(cap.dtype[name]) + ", \"calls\": [" + shapes + "]}";
    }
    std::string missing;
    for (auto & n : cap.names)
        if (!cap.data.count(n) && !cap.skipped.count(n)) missing += std::string(missing.empty() ? "" : ", ") + jstr(n);
    std::string skipped;
    for (auto & n : cap.skipped) skipped += std::string(skipped.empty() ? "" : ", ") + jstr(n);
    std::string gen;
    for (size_t i = 0; i < generated.size(); i++) gen += (i ? ", " : "") + std::to_string(generated[i]);

    man << ",\n  \"settings\": {\"n_ctx\": " << llama_n_ctx(ctx) << ", \"n_batch\": " << a.n_batch
        << ", \"n_ubatch\": " << a.n_ubatch << ", \"prompt_chunk\": " << a.prompt_chunk
        << ", \"cont_chunk\": " << a.cont_chunk << ", \"pos_offset\": " << a.pos_offset << ", \"threads\": " << cp.n_threads
        << ", \"n_gpu_layers\": " << a.n_gpu_layers << ", \"kv_type\": " << jstr(a.kv_type)
        << ", \"flash_attn\": " << jstr(a.flash_attn) << ", \"yarn_beta_fast\": " << a.yarn_beta_fast
        << ", \"yarn_beta_slow\": " << a.yarn_beta_slow << ", \"extra_bufts\": " << (a.extra_bufts ? "true" : "false")
        << ", \"split_mode\": " << jstr(a.split_mode) << ", \"main_gpu\": " << a.main_gpu << ", \"tensor_split\": [";
    for (size_t i = 0; i < a.tensor_split.size(); i++) man << (i ? ", " : "") << a.tensor_split[i];
    man << "], \"overrides\": [";
    for (size_t i = 0; i < a.overrides.size(); i++) man << (i ? ", " : "") << jstr(a.overrides[i]);
    man << "], \"greedy_tie\": \"lowest id\"}"
        << ",\n  \"n_vocab\": " << n_vocab << ",\n  \"n_embd\": " << llama_model_n_embd(model)
        << ",\n  \"n_prompt\": " << n_prompt << ",\n  \"n_tokens\": " << seq.size()
        << ",\n  \"n_scored\": " << n_scored << ",\n  \"generated\": [" << gen << "]"
        << ",\n  \"ended_by_eog\": " << (ended_by_eog ? "true" : "false")
        << ",\n  \"captures\": {" << caps << "},\n  \"captures_missing\": [" << missing << "]"
        << ",\n  \"captures_skipped\": [" << skipped << "]"
        << ",\n  \"capture_calls\": " << cap.calls
        << ",\n  \"device_memory_after_context\": [" << mem << "]"
        << ",\n  \"timing_s\": {\"load\": " << t_load << ", \"prefill\": " << t_prefill
        << ", \"decode\": " << t_decode << ", \"total\": " << t_total << "}\n}\n";
    std::ofstream(a.out + "/manifest.json") << man.str();

    llama_free(ctx);
    llama_model_free(model);
    return missing.empty() || a.captures_optional ? 0 : 3;
}
