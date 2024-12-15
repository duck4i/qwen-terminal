#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <termios.h>
#include <fcntl.h>
#include <pty.h>
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

string run_inference(llama_model *model, const string &prompt, const string &system_prompt, int n_predict = 1024)
{
    string system = "<|im_start|>system " + system_prompt + "<|im_end|>";
    string full_prompt = system + "<|im_start|>user " + prompt + "<|im_end|>" + "<|im_start|>assistant";
    string result;

    const int n_prompt = -llama_tokenize(model, full_prompt.c_str(), full_prompt.size(), NULL, 0, true, true);
    vector<llama_token> prompt_tokens(n_prompt);

    if (llama_tokenize(model, full_prompt.c_str(), full_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0)
    {
        throw runtime_error("Failed to tokenize the prompt");
    }

    // Inference
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0; // load from model itself
    ctx_params.no_perf = false;
    ctx_params.flash_attn = true;

    llama_context *ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL)
    {
        throw runtime_error("Failed to create the llama_context");
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    llama_sampler *smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    int n_decode = 0;
    llama_token new_token_id;

    int max = n_prompt + n_predict;
    for (int n_pos = 0; n_pos + batch.n_tokens < max;)
    {
        if (llama_decode(ctx, batch))
        {
            llama_sampler_free(smpl);
            llama_free(ctx);
            throw runtime_error("Failed to eval");
        }

        n_pos += batch.n_tokens;

        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_token_is_eog(model, new_token_id))
        {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0)
        {
            llama_sampler_free(smpl);
            llama_free(ctx);
            throw runtime_error("Failed to convert token to piece");
        }

        result += string(buf, n);
        batch = llama_batch_get_one(&new_token_id, 1);
        n_decode += 1;
    }

    llama_sampler_free(smpl);
    llama_free(ctx);

    return result;
}

bool write_and_execute_script(const string &script_content)
{
    const string script_path = "tmp_exec.sh";

    // Write the script to a file
    ofstream script_file(script_path);
    if (!script_file)
    {
        cerr << "Error: Could not create script file" << endl;
        return false;
    }

    script_file << "#!/bin/bash\n"
                << "set -e\n" // Exit on error
                << script_content;
    if (script_content.back() != '\n')
    {
        script_file << '\n';
    }
    script_file.close();

    // Make the script executable
    if (chmod(script_path.c_str(), S_IRWXU) != 0)
    {
        cerr << "Error: Could not make script executable" << endl;
        return false;
    }

    // Create a pseudo-terminal
    int master, slave;
    char name[1024];
    struct winsize ws;

    // Get the current terminal window size
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) < 0)
    {
        cerr << "Error: Could not get window size" << endl;
        return false;
    }

    // Create PTY with the current window size
    if (openpty(&master, &slave, name, NULL, &ws) == -1)
    {
        cerr << "Error: Failed to create pseudo-terminal" << endl;
        return false;
    }

    // Fork and execute
    pid_t pid = fork();
    if (pid == -1)
    {
        cerr << "Error: Fork failed" << endl;
        close(master);
        close(slave);
        return false;
    }

    if (pid == 0)
    { // Child process
        close(master);
        setsid();

        // Set up the slave PTY as standard streams
        dup2(slave, STDIN_FILENO);
        dup2(slave, STDOUT_FILENO);
        dup2(slave, STDERR_FILENO);

        // Get current terminfo
        const char *current_term = getenv("TERM");
        if (!current_term)
            current_term = "xterm-256color";

        // Set environment variables for color support
        setenv("TERM", current_term, 1);
        setenv("COLORTERM", "truecolor", 1);

        // Set other common color-related variables
        setenv("FORCE_COLOR", "1", 1);
        setenv("CLICOLOR", "1", 1);
        setenv("CLICOLOR_FORCE", "1", 1);

        // Close the slave side of the PTY
        close(slave);

        // Execute the script
        execl("/bin/bash", "bash", "--login", script_path.c_str(), NULL);
        exit(1);
    }
    else
    { // Parent process
        close(slave);

        // Set master FD to non-blocking mode
        int flags = fcntl(master, F_GETFL, 0);
        fcntl(master, F_SETFL, flags | O_NONBLOCK);

        char buf[4096]; // Increased buffer size for better handling of color codes
        fd_set read_fds;
        struct timeval tv;
        int status;
        bool running = true;

        // Save terminal settings
        struct termios orig_term;
        tcgetattr(STDIN_FILENO, &orig_term);

        // Get the current terminal settings
        struct termios new_term = orig_term;

        // Modify the terminal settings
        // Turn off canonical mode and echo
        new_term.c_lflag &= ~(ICANON | ECHO | ECHOE | ECHOK | ECHONL);

        // Turn on output processing for proper handling of newlines and color codes
        new_term.c_oflag |= OPOST | ONLCR;

        // Set character size
        new_term.c_cflag &= ~CSIZE;
        new_term.c_cflag |= CS8;

        // Enable reading of control characters
        new_term.c_cc[VMIN] = 1;
        new_term.c_cc[VTIME] = 0;

        // Apply the new settings
        tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

        while (running)
        {
            FD_ZERO(&read_fds);
            FD_SET(STDIN_FILENO, &read_fds);
            FD_SET(master, &read_fds);

            tv.tv_sec = 0;
            tv.tv_usec = 100000; // 100ms timeout

            int ret = select(master + 1, &read_fds, NULL, NULL, &tv);

            if (ret > 0)
            {
                // Check for input from user
                if (FD_ISSET(STDIN_FILENO, &read_fds))
                {
                    int n = read(STDIN_FILENO, buf, sizeof(buf));
                    if (n > 0)
                    {
                        write(master, buf, n);
                    }
                }

                // Check for output from script
                if (FD_ISSET(master, &read_fds))
                {
                    int n = read(master, buf, sizeof(buf));
                    if (n > 0)
                    {
                        // Write directly to STDOUT_FILENO to preserve color codes
                        write(STDOUT_FILENO, buf, n);
                        fflush(stdout);
                    }
                }
            }

            // Handle window size changes
            struct winsize new_ws;
            if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &new_ws) == 0)
            {
                if (memcmp(&ws, &new_ws, sizeof(struct winsize)) != 0)
                {
                    ws = new_ws;
                    ioctl(master, TIOCSWINSZ, &ws);
                }
            }

            // Check if child process has ended
            pid_t wpid = waitpid(pid, &status, WNOHANG);
            if (wpid > 0)
            {
                // Read any remaining output
                while (true)
                {
                    int n = read(master, buf, sizeof(buf));
                    if (n <= 0)
                        break;
                    write(STDOUT_FILENO, buf, n);
                    fflush(stdout);
                }
                running = false;
            }
        }

        // Restore original terminal settings
        tcsetattr(STDIN_FILENO, TCSANOW, &orig_term);

        // Clean up the temporary script file
        if (remove(script_path.c_str()) != 0)
        {
            cerr << "Warning: Could not remove temporary script file" << endl;
        }

        close(master);

        if (WIFEXITED(status))
        {
            return WEXITSTATUS(status) == 0;
        }
        return false;
    }
}

std::string extract_bash_content(const std::string &input)
{
    const std::string start_marker = "```bash\n";
    const std::string end_marker = "\n```";

    size_t start_pos = input.find(start_marker);
    if (start_pos == std::string::npos)
        return input;

    size_t content_start = start_pos + start_marker.length();
    size_t content_end = input.find(end_marker, content_start);

    if (content_end == std::string::npos)
        return input;

    return input.substr(content_start, content_end - content_start);
}

int main(int argc, char *argv[])
{
    int n_log;
    int n_predict = 1024; // number of tokens to predict
    string s_system = R"(You are a bash script generator that produces only executable shell commands. Your output must be a complete, ready-to-run bash script that begins with '#!/bin/bash' and contains only valid bash syntax. 

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

        Remember: Generate only the final, executable script - no explanations, no examples, just working code.)";

    OptionParser parser("Allowed options");
    auto help_option = parser.add<Switch>("h", "help", "Produce help message.");
    auto print_system_info = parser.add<Switch>("d", "sys_info", "Prints build system info.");
    auto max_token_option = parser.add<Value<int>>("t", "tokens", "Max token count for output (length of response).", 1024, &n_predict);
    auto log_option = parser.add<Value<int>>("l", "log", "Log level. 0 - none, 1 - debug, 2 - info, 3 - warn, 4 - error", (int)GGML_LOG_LEVEL_WARN, &n_log);
    auto model_option = parser.add<Value<string>>("m", "model", "Filename of model to load.", "qwen2.5-0.5b-instruct-fp16.gguf");
    auto system_option = parser.add<Value<string>>("s", "system", "System message for the model.", s_system, &s_system);

    parser.parse(argc, argv);

    if (print_system_info.get()->is_set())
    {
        cout << llama_print_system_info() << endl;
        return 0;
    }

    if (help_option.get()->is_set())
    {
        cout << parser << endl;
        return 0;
    }

    ggml_backend_load_all();
    llama_log_set(log, (void *)&n_log);

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.main_gpu = 0;

    llama_model *model = llama_load_model_from_file(model_option.get()->value().c_str(), model_params);

    if (model == NULL)
    {
        cerr << "Error: Unable to load model" << endl;
        return 1;
    }

    try
    {
        string command;

        while (true)
        {
            cout << "qwterm> ";
            getline(cin, command);

            if (command == "exit")
            {
                break; // Exit the shell
            }

            string result = run_inference(
                model,
                command,
                s_system,
                n_predict);

            result = extract_bash_content(result);
            if (n_log == GGML_LOG_LEVEL_INFO)
                cout << result << endl;

            bool success = write_and_execute_script(result);
            if (!success)
            {
                cerr << "Script execution failed" << endl;
            }
        }
    }
    catch (const runtime_error &e)
    {
        cerr << "Error: " << e.what() << endl;
    }

    cout << "Exiting.." << endl;

    // Clean up model at the end
    llama_free_model(model);
    return 0;
}