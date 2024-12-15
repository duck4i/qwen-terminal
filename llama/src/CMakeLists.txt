function(llama_add_compile_flags)
    if (LLAMA_FATAL_WARNINGS)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(APPEND C_FLAGS   -Werror)
            list(APPEND CXX_FLAGS -Werror)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            add_compile_options(/WX)
        endif()
    endif()

    if (LLAMA_ALL_WARNINGS)
        if (NOT MSVC)
            list(APPEND C_FLAGS -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                -Werror=implicit-int -Werror=implicit-function-declaration)

            list(APPEND CXX_FLAGS -Wmissing-declarations -Wmissing-noreturn)

            list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)

            list(APPEND C_FLAGS   ${WARNING_FLAGS})
            list(APPEND CXX_FLAGS ${WARNING_FLAGS})

            ggml_get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

            add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                                "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
        else()
            # todo : msvc
            set(C_FLAGS   "" PARENT_SCOPE)
            set(CXX_FLAGS "" PARENT_SCOPE)
        endif()
    endif()
endfunction()

llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-vocab.cpp
            llama-grammar.cpp
            llama-sampling.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            )

target_include_directories(llama PUBLIC . ../include)
target_compile_features   (llama PUBLIC cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
