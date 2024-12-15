# Qwen LLM terminal 

A sub terminal where you tell it what you want and it executes bash commands for you.

It works with any models but I was testing small Qwen-2.5 0.5B.

## Options

```
Allowed options:
  -h, --help                          Produce help message.
  -d, --sys_info                      Prints build system info.
  -t, --tokens arg (=1024)            Max token count for output (length of response).
  -l, --log arg (=3)                  Log level. 0 - none, 1 - debug, 2 - info, 3 - warn, 4 - error
  -m, --model arg (=qwen2.5-0.5b-instruct-fp16.gguf)
                                      Filename of model to load.
  -s, --system arg (=You are a bash script generator that produces only executable shell commands. Your output must be a complete, ready-to-run bash script that begins with '#!/bin/bash' and contains only valid bash syntax. 

        Never include:
        - Function definitions or example usage
        - Comments or explanations
        - Placeholders or pseudo-code

        Always include:
        - Error handling with appropriate exit codes
        - Input validation where necessary
        - Status messages using 'echo' to inform about success/failure
        - Required shell environment checks if needed

        The script should execute commands sequentially in the current directory unless explicitly specified otherwise. Every command must be properly escaped and use absolute paths when necessary.

        Use -e flag to ensure the script stops on first error:
        set -e

        Remember: Generate only the final, executable script - no explanations, no examples, just working code.)
                                      System message for the model.


```

## GGUFs
* https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true