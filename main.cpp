#include <iostream>
#include <string>
#include <vector>
#include "llama-cpp.h"
#include "popl.hpp"

using namespace std;
using namespace popl;

void log(ggml_log_level level, const char *text, void *data)
{
    int specified_level = *(static_cast<int *>(data));
    if ((level >= specified_level && level != GGML_LOG_LEVEL_CONT) && text != nullptr)
        printf("%s", text);
}

int main(int argc, char *argv[])
{
    int n_log;
    int n_predict; // number of tokens to predict
    string s_system = "You are a linux command assistant, and you provide a bash script with shell commands for any requests that user demands."
                      "Think carefully, and provide only the needed commands with recommended arguments without explanation, the desired output is a working and executable bash file."
                      "Double check every line in the bash file for validity."
                      //"If a single command is enough to complete the request, only output that command."
                      //"If there are more then a single command needed, make sure each command is in its own new line."
                      ;

    OptionParser parser("Allowed options");
    auto help_option = parser.add<Switch>("h", "help", "Produce help message.");
    auto print_system_info = parser.add<Switch>("d", "sys_info", "Prints build system info.");
    auto max_token_option = parser.add<Value<int>>("t", "tokens", "Max token count for output (lenght of response).", 1024, &n_predict);
    auto log_option = parser.add<Value<int>>("l", "log", "Log level. 0 - none, 1 - debug, 2 - info, 3 - warn, 4 - error", (int)GGML_LOG_LEVEL_WARN, &n_log);
    auto prompt_option = parser.add<Value<string>>("p", "prompt", "Your actual prompt for LLM (please use quotes).");
    auto model_option = parser.add<Value<string>>("m", "model", "Filename of model to load.", "qwen2.5-0.5b-instruct-fp16.gguf");
    auto system_option = parser.add<Value<string>>("s", "system", "System message for the model.", s_system, &s_system);

    parser.parse(argc, argv);

    bool prompt_is_set = prompt_option.get()->is_set();

    if (print_system_info.get()->is_set())
    {
        cout << llama_print_system_info() << endl;
        return prompt_is_set ? 0 : 1;
    }

    if (help_option.get()->is_set() || !prompt_is_set)
    {
        cout << parser << endl;
        return prompt_is_set ? 0 : 1;
    }

    auto prompt = prompt_option.get()->value();

    string system = "<|im_start|>system " + s_system + "<|im_end|>";
    string full_prompt = system + "<|im_start|>user " + prompt + "<|im_end|>" + "<|im_start|>assistant";

    ggml_backend_load_all();
    llama_log_set(log, (void *)&n_log);

    //  Load model
    llama_model_params model_params = llama_model_default_params();

    llama_model *model = llama_load_model_from_file(model_option.get()->value().c_str(), model_params);
    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_prompt = -llama_tokenize(model, full_prompt.c_str(), full_prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(model, full_prompt.c_str(), full_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0)
    {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    //  Inference part
    {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 0; // load from model iteself
        ctx_params.no_perf = false;
        ctx_params.flash_attn = true;

        llama_context *ctx = llama_new_context_with_model(model, ctx_params);

        if (ctx == NULL)
        {
            fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
            return 1;
        }

        // initialize the sampler

        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;

        llama_sampler *smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // prepare a batch for the prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        // main loop
        int n_decode = 0;
        llama_token new_token_id;

        int max = n_prompt + n_predict;
        for (int n_pos = 0; n_pos + batch.n_tokens < max;)
        {
            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch))
            {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }

            n_pos += batch.n_tokens;

            // sample the next token
            {
                new_token_id = llama_sampler_sample(smpl, ctx, -1);

                // is it an end of generation?
                if (llama_token_is_eog(model, new_token_id))
                {
                    break;
                }

                char buf[128];
                int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
                if (n < 0)
                {
                    fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                    return 1;
                }

                string s(buf, n);
                printf("%s", s.c_str());
                fflush(stdout);

                // prepare the next batch with the sampled token
                batch = llama_batch_get_one(&new_token_id, 1);

                n_decode += 1;
            }
        }

        llama_sampler_free(smpl);
        llama_free(ctx);
    }

    printf("\n");
    llama_free_model(model);

    return 0;
}