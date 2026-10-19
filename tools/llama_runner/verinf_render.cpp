// A frozen passage's prompt as a llama.cpp deployment would build it
// (analysis/k2-fidelity-study-design.md §6.4): the model's own chat template,
// applied by llama.cpp's Jinja engine to one user message with the assistant
// generation prefix, then tokenized as llama-completion does (special tokens
// added and parsed). The model is loaded vocabulary-only; a metadata-only GGUF
// with the pinned file's keys is enough.
//
//   verinf_render --model M --content user.txt --out DIR
// writes DIR/prompt.txt (the rendered text), DIR/ids.txt and DIR/render.json.
#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#ifndef VERINF_LLAMA_COMMIT
#define VERINF_LLAMA_COMMIT "unknown"
#endif

static std::string jstr(const std::string & s) {
    std::string o = "\"";
    for (char ch : s) {
        if (ch == '"' || ch == '\\') { o += '\\'; o += ch; }
        else if (ch == '\n') o += "\\n";
        else if ((unsigned char) ch < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", ch); o += b; }
        else o += ch;
    }
    return o + "\"";
}

int main(int argc, char ** argv) {
    std::string model_path, content_path, out;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--model") model_path = v;
        else if (k == "--content") content_path = v;
        else if (k == "--out") out = v;
        else { fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (model_path.empty() || content_path.empty() || out.empty()) {
        fprintf(stderr, "usage: verinf_render --model M --content FILE --out DIR\n");
        return 2;
    }
    std::ifstream cf(content_path, std::ios::binary);
    std::stringstream ss;
    ss << cf.rdbuf();
    const std::string content = ss.str();

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs in;
    in.use_jinja = true;
    in.add_generation_prompt = true;
    common_chat_msg msg;
    msg.role = "user";
    msg.content = content;
    in.messages = {msg};
    const auto params = common_chat_templates_apply(tmpls.get(), in);
    const std::string prompt = params.prompt.str();
    const auto ids = common_tokenize(vocab, prompt, true, true);
    // the user's text alone, special tokens not parsed: if parsing changed it, the
    // content holds special-token text and the whole-string tokenization would read it
    const auto content_plain = common_tokenize(vocab, content, false, false);
    const auto content_special = common_tokenize(vocab, content, false, true);

    std::ofstream(out + "/prompt.txt", std::ios::binary) << prompt;
    std::ofstream idf(out + "/ids.txt");
    for (auto id : ids) idf << id << "\n";
    std::string parts;
    for (const auto & p : params.prompt.parts)
        parts += std::string(parts.empty() ? "" : ", ") + "{\"is_input\": " + (p.is_input ? "true" : "false") +
                 ", \"chars\": " + std::to_string(p.text.size()) + "}";
    std::ofstream(out + "/render.json")
        << "{\n  \"llama_cpp_commit\": " << jstr(VERINF_LLAMA_COMMIT)
        << ",\n  \"model\": " << jstr(model_path)
        << ",\n  \"template\": " << jstr(common_chat_templates_source(tmpls.get()))
        << ",\n  \"add_bos\": " << (llama_vocab_get_add_bos(vocab) ? "true" : "false")
        << ",\n  \"n_tokens\": " << ids.size()
        << ",\n  \"content_has_special_tokens\": " << (content_plain != content_special ? "true" : "false")
        << ",\n  \"parts\": [" << parts << "]\n}\n";
    llama_model_free(model);
    return 0;
}
