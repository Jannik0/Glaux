# llama.cpp and transcribe.cpp keep ggml's default GGML_MAX_NAME of 64.
# stable-diffusion.cpp compiles with 160 because diffusion tensor names are
# longer, and that changes the size of ggml_tensor. Defining it here lets
# those two engines load stable-diffusion.cpp's ggml-cuda module.
add_compile_definitions(GGML_MAX_NAME=160)
