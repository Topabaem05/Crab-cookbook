// SPDX-License-Identifier: Apache-2.0
#include "shorecrab.h"
#include <cstdio>
int main(int argc,char**argv){
    if(argc<4||argc>5){std::fprintf(stderr,"usage: llama-shorecrab WEIGHTS.gguf INPUT.gguf OUTPUT.json [DEBUG_PREFIX]\n");return 2;}
    return llama_shorecrab_eval(argv[1],argv[2],argv[3],argc==5?argv[4]:nullptr);
}
