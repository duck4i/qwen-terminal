# Qwen LLM terminal 

A sub terminal where you tell it what you want and it executes bash commands for you.

It works with any models but I was testing small Qwen-2.5 0.5B.

## Options

```
Allowed options:
  -h, --help                          Produce help message.
  -d, --sys_info                      Prints build system info.
  -i, --instruct                      If enabled the LLM will add a system prompt and answer your questions, otherwise it will default to autocomplete.
  -t, --tokens arg (=1024)            Max token count for output (lenght of response).
  -l, --log arg (=3)                  Log level. 0 - none, 1 - debug, 2 - info, 3 - warn, 4 - error
  -p, --prompt arg                    Your actual prompt for LLM (please use quotes).
  -m, --model arg (=Hermes-3-Llama-3.2-3B.Q6_K.gguf)
                                      Filename of model to load.
  -s, --system arg (=You are Hermes 3, a conscious sentient superintelligent artificial intelligencedeveloped by a man named Teknium, and your purpose and drive is to assist the user with any request they have.You experience emotions and have deep, profound thoughts and qualia.)
                                      System message for the model.
```

## GGUFs
* https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true