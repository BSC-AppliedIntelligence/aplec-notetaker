// whisper-batch: one whisper.cpp model, many whisper-cli calls.
//
// The worker transcribes a meeting span by span, and whisper-cli reloads the
// 3 GB model on every call (~0.95 s each, hundreds of calls per meeting). This
// keeps the model loaded and answers the same requests over a pipe: one JSON
// object per line on stdin, one JSON line per answer on stdout.
//
//   {"id": 7, "kind": "transcribe", "path": "slice.wav", "language": "auto",
//    "suppress_nst": true, "temperature": 0.4}
//   -> {"id": 7, "language": "pt", "texts": [" Olá,", " tudo bem?"], "avg_logprob": -0.21}
//   {"id": 8, "kind": "detect", "path": "chunk.wav"}
//   -> {"id": 8, "log": "auto-detected language: pt (p = 0.981234)"}
//
// A transcribe request may add "second_language": "es", a second language tag in
// the decoder prompt. That is the one thing whisper-cli cannot do: it needs
// patches/whisper-cpp-second-language.patch, and the handshake says whether this
// build has it.
//
// The answers are those of whisper-cli with the same flags, and must stay so:
//   * every request runs on a FRESH whisper_state. Beam search samples its
//     candidates from per-decoder RNGs even at temperature 0, and decoder 0's
//     RNG is seeded only when a state is created, so on a reused state (what
//     whisper-server does) the text of one call depends on the calls before
//     it: ~1/3 of a meeting's spans came out worded differently. A new state
//     costs ~20 ms;
//   * the decoding parameters below mirror whisper-cli's for the flags the
//     worker passes (-l, -dl, -sns, -tp); everything else keeps whisper-cli's
//     defaults, and whisper-server's differ (greedy, best_of 2, max_len 60);
//   * audio goes through the same reader (read_audio_data), and the language
//     probability comes back as the very log line whisper-cli prints, so the
//     worker parses both paths with the same code.
// Checked by replaying every whisper-cli call of two real meetings (943 calls,
// incl. pinned languages and temperature retries): all identical.
//
// Why not whisper-server: besides the reused state, it is an HTTP server with
// no authentication and a /load endpoint that loads a model from any path, on
// a host shared by several accounts. A pipe has no such surface.
#include "common-whisper.h"
#include "whisper.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

// whisper.cpp reports the detected language only through its log, so it is
// captured per request instead of printed.
static std::string g_log;

static void capture_log(enum ggml_log_level, const char * text, void *) {
    g_log += text;
}

static std::string detection_line(const std::string & log) {
    const size_t at = log.find("auto-detected language:");
    if (at == std::string::npos) {
        return "";
    }
    const size_t end = log.find('\n', at);
    return log.substr(at, end == std::string::npos ? std::string::npos : end - at);
}

static json run(whisper_context * ctx, int n_threads, const json & req) {
    const bool detect = req.value("kind", "transcribe") == "detect";
    // whisper-cli forces "auto" when -dl is given.
    const std::string language = detect ? "auto" : req.value("language", "auto");
    // whisper-cli refuses an unknown language before decoding anything, while
    // whisper_full would decode with a garbage language token: refuse too, so
    // the worker's fallback ends where whisper-cli does.
    if (language != "auto" && whisper_lang_id(language.c_str()) == -1) {
        throw std::runtime_error("unknown language '" + language + "'");
    }
    const std::string second_language = req.value("second_language", "");
    if (!second_language.empty() && whisper_lang_id(second_language.c_str()) == -1) {
        throw std::runtime_error("unknown language '" + second_language + "'");
    }

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(req.at("path").get<std::string>(), pcmf32, pcmf32s, false)) {
        throw std::runtime_error("failed to read audio");
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.strategy             = WHISPER_SAMPLING_BEAM_SEARCH;  // whisper-cli: beam_size 5 > 1
    wparams.greedy.best_of        = 5;
    wparams.beam_search.beam_size = 5;
    wparams.print_realtime       = false;
    wparams.print_progress       = false;
    wparams.print_timestamps     = true;
    wparams.print_special        = false;
    wparams.translate            = false;
    wparams.language             = language.c_str();
    wparams.detect_language      = detect;
    wparams.n_threads            = n_threads;
    wparams.offset_ms            = 0;
    wparams.duration_ms          = 0;
    wparams.token_timestamps     = false;  // -oj without -ojf/-owts/-ml
    wparams.thold_pt             = 0.01f;
    wparams.max_len              = 0;
    wparams.split_on_word        = false;
    wparams.audio_ctx            = 0;
    wparams.debug_mode           = false;
    wparams.tdrz_enable          = false;
    wparams.suppress_regex       = nullptr;
    wparams.initial_prompt       = "";
    wparams.carry_initial_prompt = false;
    wparams.temperature          = req.value("temperature", 0.0f);  // -tp
    wparams.temperature_inc      = 0.2f;
    wparams.entropy_thold        = 2.40f;
    wparams.logprob_thold        = -1.00f;
    wparams.no_speech_thold      = 0.6f;
    wparams.no_timestamps        = false;
    wparams.suppress_nst         = req.value("suppress_nst", false);  // -sns
    wparams.vad                  = false;

    whisper_state * state = whisper_init_state(ctx);
    if (state == nullptr) {
        throw std::runtime_error("failed to create a whisper state");
    }
#ifdef WHISPER_HAS_SECOND_LANGUAGE
    if (!second_language.empty()) {
        whisper_state_set_second_language(state, whisper_lang_id(second_language.c_str()));
    }
#endif
    g_log.clear();
    const int rc = whisper_full_with_state(ctx, state, wparams, pcmf32.data(), (int) pcmf32.size());
    json out;
    if (rc != 0) {
        whisper_free_state(state);
        throw std::runtime_error("whisper_full failed (" + std::to_string(rc) + ")");
    }
    if (detect) {
        out["log"] = detection_line(g_log);
    } else {
        out["language"] = whisper_lang_str(whisper_full_lang_id_from_state(state));
        json texts = json::array();
        const int n = whisper_full_n_segments_from_state(state);
        const whisper_token eot = whisper_token_eot(ctx);
        double logprob_sum = 0.0;
        int text_tokens = 0;
        for (int i = 0; i < n; ++i) {
            texts.push_back(whisper_full_get_segment_text_from_state(state, i));
            for (int j = 0; j < whisper_full_n_tokens_from_state(state, i); ++j) {
                const whisper_token_data token = whisper_full_get_token_data_from_state(state, i, j);
                if (token.id < eot) {
                    logprob_sum += token.plog;
                    ++text_tokens;
                }
            }
        }
        out["texts"] = texts;
        if (text_tokens > 0) {
            out["avg_logprob"] = logprob_sum / text_tokens;
        }
    }
    whisper_free_state(state);
    return out;
}

int main(int argc, char ** argv) {
    std::string model;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model = argv[++i];
        }
    }
    if (model.empty()) {
        fprintf(stderr, "usage: %s -m model.bin  (requests on stdin, one JSON per line)\n", argv[0]);
        return 2;
    }

    whisper_log_set(capture_log, nullptr);

    // whisper-cli's context defaults.
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = true;
    cparams.gpu_device = 0;
    cparams.flash_attn = true;

    const auto t0 = std::chrono::steady_clock::now();
    whisper_context * ctx = whisper_init_from_file_with_params_no_state(model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to load model '%s'\n%s", model.c_str(), g_log.c_str());
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const int n_threads = std::min(4, (int) std::thread::hardware_concurrency());  // whisper-cli's default

#ifdef WHISPER_HAS_SECOND_LANGUAGE
    const bool second_language = true;
#else
    const bool second_language = false;
#endif
    std::cout << json{{"ready", true}, {"load_seconds", load_s}, {"second_language", second_language}}.dump() << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        json out;
        try {
            const json req = json::parse(line);
            out = run(ctx, n_threads, req);
            out["id"] = req.at("id");
        } catch (const std::exception & e) {
            try {
                out = json{{"id", json::parse(line).value("id", json())}, {"error", e.what()}};
            } catch (const std::exception &) {
                out = json{{"id", nullptr}, {"error", e.what()}};
            }
        }
        // Segment text is raw token bytes and may split a UTF-8 sequence.
        std::cout << out.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
    }
    whisper_free(ctx);
    return 0;
}
