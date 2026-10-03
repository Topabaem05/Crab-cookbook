// SPDX-License-Identifier: Apache-2.0
// A60 inference graph on the pinned llama.cpp GGML CPU backend.
#include "shorecrab.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
using T=ggml_tensor;
using V=std::vector<float>;
struct File {
    ggml_context *ctx=nullptr;gguf_context *meta=nullptr;
    explicit File(const char *path){meta=gguf_init_from_file(path,{false,&ctx});if(!meta||!ctx)throw std::runtime_error("cannot read GGUF");}
    ~File(){if(meta)gguf_free(meta);if(ctx)ggml_free(ctx);}
    T *get(const std::string &name){auto *t=ggml_get_tensor(ctx,name.c_str());if(!t)throw std::runtime_error("missing tensor: "+name);return t;}
};
struct Graph {
    ggml_context *c;ggml_backend_t backend;ggml_gallocr_t allocator;
    std::vector<std::pair<T*,V>> inputs;
    Graph(){c=ggml_init({32*1024*1024,nullptr,true});backend=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(backend,2);allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));}
    ~Graph(){ggml_gallocr_free(allocator);ggml_backend_free(backend);ggml_free(c);}
    T *input(const V &v,int64_t d,int64_t n=1){auto*t=ggml_new_tensor_2d(c,GGML_TYPE_F32,d,n);if(ggml_nelements(t)!=(int64_t)v.size())throw std::runtime_error("input shape mismatch");ggml_set_input(t);inputs.push_back({t,v});return t;}
    V run(T *out){out=ggml_cont(c,out);ggml_set_output(out);auto*g=ggml_new_graph_custom(c,16000,false);ggml_build_forward_expand(g,out);if(!ggml_gallocr_alloc_graph(allocator,g))throw std::runtime_error("graph allocation failed");
        for(auto &p:inputs)ggml_backend_tensor_set(p.first,p.second.data(),0,p.second.size()*4);
        if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("GGML computation failed");V v(ggml_nelements(out));ggml_backend_tensor_get(out,v.data(),0,v.size()*4);return v;}
    T *add(T*a,T*b){return ggml_add(c,a,b);}T *mul(T*a,T*b){return ggml_mul(c,a,b);}
    T *cont(T*a){return ggml_cont(c,a);}T *gelu(T*a){return ggml_gelu_erf(c,a);}
    T *linear(File&w,T*x,const std::string&p){return add(ggml_mul_mat(c,w.get(p+".weight"),x),w.get(p+".bias"));}
    T *norm(File&w,T*x,const std::string&p,float eps=1e-5){return add(mul(ggml_norm(c,x,eps),w.get(p+".weight")),w.get(p+".bias"));}
    T *attention(T*q,T*k,T*v,int heads,T*bias=nullptr){int d=q->ne[0],n=q->ne[1],m=k->ne[1];
        q=cont(ggml_permute(c,ggml_reshape_3d(c,q,d/heads,heads,n),0,2,1,3));
        k=cont(ggml_permute(c,ggml_reshape_3d(c,k,d/heads,heads,m),0,2,1,3));
        v=cont(ggml_permute(c,ggml_reshape_3d(c,v,d/heads,heads,m),0,2,1,3));
        auto*s=ggml_scale(c,ggml_mul_mat(c,k,q),1/std::sqrt(float(d/heads)));if(bias)s=add(s,bias);
        auto*p=ggml_soft_max(c,s);auto*z=ggml_mul_mat(c,cont(ggml_transpose(c,v)),p);
        return ggml_reshape_2d(c,cont(ggml_permute(c,z,0,2,1,3)),d,n);}
    T *cross(File&w,T*q,T*m,const std::string&p,T*bias=nullptr){return linear(w,attention(linear(w,q,p+".q"),linear(w,m,p+".k"),linear(w,m,p+".v"),6,bias),p+".out");}
    T *ffn(File&w,T*x,const std::string&p){return add(x,linear(w,gelu(linear(w,norm(w,x,p+".net.0"),p+".net.1")),p+".net.3"));}
    T *conv_raw(T*a,T*b,int stride,int pad,bool depthwise){
        T*cols;T*r;
        if(depthwise){
            auto*aa=ggml_reshape_4d(c,a,a->ne[0],a->ne[1],1,a->ne[3]);
            cols=ggml_im2col(c,aa,ggml_reshape_4d(c,b,b->ne[0],b->ne[1],1,b->ne[2]),stride,stride,pad,pad,1,1,true,GGML_TYPE_F32);
            auto*bb=ggml_reshape_4d(c,cols,cols->ne[0],cols->ne[1]*cols->ne[2],b->ne[2],1);
            aa=ggml_reshape_4d(c,aa,a->ne[0]*a->ne[1],1,a->ne[3],1);
            r=ggml_mul_mat(c,aa,bb);return ggml_reshape_4d(c,r,cols->ne[1],cols->ne[2],b->ne[2],1);
        }
        cols=ggml_im2col(c,a,b,stride,stride,pad,pad,1,1,true,GGML_TYPE_F32);
        r=ggml_mul_mat(c,ggml_reshape_2d(c,cols,cols->ne[0],cols->ne[1]*cols->ne[2]),ggml_reshape_2d(c,a,a->ne[0]*a->ne[1]*a->ne[2],a->ne[3]));
        return ggml_reshape_4d(c,r,cols->ne[1],cols->ne[2],a->ne[3],1);
    }
    T *channel(T*x,T*v){return ggml_reshape_4d(c,v,1,1,x->ne[2],1);}
    T *conv(File&w,T*x,const std::string&p,int stride=1,int groups=1,bool bias=true){auto*weight=w.get(p+".weight");int pad=weight->ne[0]/2;T*y;
        if(groups>1&&weight->ne[3]==2*x->ne[2]){
            auto*a=conv_raw(w.get(p+".weight.even"),x,stride,pad,true);auto*b=conv_raw(w.get(p+".weight.odd"),x,stride,pad,true);
            auto*z=ggml_concat(c,a,b,3);z=cont(ggml_permute(c,z,0,1,3,2));y=ggml_reshape_4d(c,z,a->ne[0],a->ne[1],weight->ne[3],1);
        }else y=conv_raw(weight,x,stride,pad,groups>1);
        return bias?add(y,channel(y,w.get(p+".bias"))):y;
    }
};
V vision(File&w,T*pixels,int index){Graph g;auto*c=g.c;const int64_t stride=512*512*3;
    auto*x=ggml_view_4d(c,pixels,512,512,3,1,512*4,512*512*4,stride*4,index*stride*4);
    for(int i=0;i<3;i++)x=g.gelu(g.conv(w,x,"vision.stem."+std::to_string(i)+".reparam_conv",i<2?2:1,i==1?48:1));
    T*f16=nullptr;int layers[]={2,2,4,2};
    for(int si=0;si<4;si++){
        auto pre="vision.stages."+std::to_string(si)+".base";
        if(si){x=g.conv(w,x,pre+".downsample.proj.0.reparam_conv",2,x->ne[2]);x=g.gelu(g.conv(w,x,pre+".downsample.proj.1.reparam_conv"));}
        for(int bi=0;bi<layers[si];bi++){
            auto p=pre+".blocks."+std::to_string(bi);x=g.conv(w,x,p+".token_mixer.reparam_conv",1,x->ne[2]);
            auto*y=g.conv(w,x,p+".mlp.conv.conv",1,x->ne[2],false);
            y=g.add(g.mul(y,g.channel(y,w.get(p+".mlp.conv.gain"))),g.channel(y,w.get(p+".mlp.conv.offset")));
            y=g.conv(w,g.gelu(g.conv(w,y,p+".mlp.fc1")),p+".mlp.fc2");
            y=g.mul(y,g.channel(y,w.get(p+".layer_scale.gamma")));x=g.add(x,y);
        }
        if(si==2)f16=x;
    }
    auto*a=g.conv(w,f16,"project16");auto*b=g.conv(w,x,"project32");
    b=ggml_upscale(c,b,2,GGML_SCALE_MODE_BILINEAR);auto*f=g.add(a,g.mul(b,w.get("scale32")));
    auto*t=g.cont(ggml_transpose(c,ggml_reshape_2d(c,f,1024,384)));
    return g.run(g.norm(w,t,"visual_norm"));
}
V bert(File&w,const std::vector<int32_t>&ids){Graph g;auto*c=g.c;int n=ids.size();
    // Index tensors are read-only host inputs; lifetime exceeds graph execution.
    ggml_context*ic=ggml_init({4096+size_t(n)*16,nullptr,false});
    auto*ti=ggml_new_tensor_1d(ic,GGML_TYPE_I32,n);memcpy(ti->data,ids.data(),n*4);
    auto*pi=ggml_new_tensor_1d(ic,GGML_TYPE_I32,n);std::iota((int32_t*)pi->data,(int32_t*)pi->data+n,0);
    std::string e="text.encoder.embeddings";
    auto*x=g.add(ggml_get_rows(c,w.get(e+".word_embeddings.weight"),ti),ggml_get_rows(c,w.get(e+".position_embeddings.weight"),pi));
    auto*type=ggml_view_2d(c,w.get(e+".token_type_embeddings.weight"),384,1,384*4,0);x=g.norm(w,g.add(x,type),e+".LayerNorm",1e-12f);
    for(int i=0;i<12;i++){
        auto p="text.encoder.encoder.layer."+std::to_string(i);auto s=p+".attention.self";
        auto*z=g.attention(g.linear(w,x,s+".query"),g.linear(w,x,s+".key"),g.linear(w,x,s+".value"),12);
        x=g.norm(w,g.add(x,g.linear(w,z,p+".attention.output.dense")),p+".attention.output.LayerNorm",1e-12f);
        z=g.linear(w,g.gelu(g.linear(w,x,p+".intermediate.dense")),p+".output.dense");
        x=g.norm(w,g.add(x,z),p+".output.LayerNorm",1e-12f);
    }
    auto out=g.run(x);ggml_free(ic);return out;
}
V mean(const V&x){V out(384,0);int n=x.size()/384;for(int i=0;i<n;i++)for(int d=0;d<384;d++)out[d]+=x[i*384+d]/n;return out;}
V resample(File&w,const V&tokens,const V&qcls,const float*geom,const float*weights,int n){Graph g;auto*c=g.c;V mem=tokens,bias(1024);for(size_t j=0;j<mem.size();j++)mem[j]+=geom[j];for(int j=0;j<1024;j++)bias[j]=weights[j]>0?std::log(std::max(weights[j],1e-12f)):-INFINITY;
    auto*bank=ggml_view_2d(c,w.get("resampler.bank"),384,n,384*4,0);
    auto*q=g.add(bank,g.linear(w,g.input(qcls,384),"resampler.qcond"));
    auto*z=g.cross(w,g.norm(w,q,"resampler.qnorm"),g.norm(w,g.input(mem,384,1024),"resampler.mnorm"),"resampler.attention",g.input(bias,1024));
    z=g.add(bank,z);return g.run(g.norm(w,g.ffn(w,z,"resampler.ffn"),"resampler.norm"));
}
V fuse_choice(File&w,const V&a,const V&memory){Graph g;auto*c=g.c;auto*x=g.input(a,384,a.size()/384);auto*m=g.input(memory,384,memory.size()/384);
    for(int i=0;i<2;i++){auto p="fusion."+std::to_string(i);auto*y=g.norm(w,x,p+".norm");auto s=p+".self_attn";
        auto*z=g.attention(g.linear(w,y,s+".q"),g.linear(w,y,s+".k"),g.linear(w,y,s+".v"),6);
        x=g.add(x,g.linear(w,z,s+".out_proj"));x=g.add(x,g.cross(w,g.norm(w,x,p+".qnorm"),g.norm(w,m,p+".mnorm"),p+".cross"));x=g.ffn(w,x,p+".ffn");}
    return g.run(ggml_view_2d(c,x,384,1,x->nb[1],0));
}
V score(File&w,const V&hs,int k){Graph g;auto*x=g.input(hs,384,k);return g.run(g.linear(w,g.gelu(g.linear(w,g.norm(w,x,"scorer.0"),"scorer.1")),"scorer.3"));}
float answerability(File&w,const V&q,const V&summary,const V&hs,int k){V joined=q;joined.insert(joined.end(),summary.begin(),summary.end());auto avg=mean(hs);joined.insert(joined.end(),avg.begin(),avg.end());for(int d=0;d<384;d++){float m=-INFINITY;for(int i=0;i<k;i++)m=std::max(m,hs[i*384+d]);joined.push_back(m);}joined.push_back(std::log(float(k)));Graph g;return g.run(g.linear(w,g.gelu(g.linear(w,g.input(joined,1537),"answerability.0")),"answerability.2"))[0];}
void array(std::ostream&o,const V&v){o<<'[';for(size_t i=0;i<v.size();i++){if(i)o<<',';o<<v[i];}o<<']';}
void dump(const std::string&path,const std::string&name,const V&v){if(path.empty())return;std::ofstream f(path+"."+name+".f32",std::ios::binary);f.write((const char*)v.data(),v.size()*4);}
int llama_shorecrab_eval(const char* model_path,const char* input_path,const char* output_path,const char* debug_prefix){try{
    if(!model_path||!input_path||!output_path)throw std::runtime_error("missing path");
    File w(model_path),in(input_path);std::string debug=debug_prefix?debug_prefix:"";
    auto fmt=gguf_find_key(w.meta,"shorecrab.format_version");if(fmt<0||gguf_get_val_u32(w.meta,fmt)!=1)throw std::runtime_error("unsupported A60 GGUF");
    auto*pixel=in.get("pixels");int views=pixel->ne[3];if(pixel->ne[0]!=512||pixel->ne[1]!=512||pixel->ne[2]!=3||views<1||views>257)throw std::runtime_error("invalid image views");
    auto*qi=in.get("question_ids");auto*ai=in.get("choice_ids");auto*mask=in.get("choice_mask");int k=ai->ne[1];if(k<2||k>16||qi->ne[0]>96||ai->ne[0]>48)throw std::runtime_error("invalid text shape");
    auto*qm=in.get("question_mask");auto*ge=in.get("geometry");auto*ow=in.get("owned_weights");
    if(pixel->type!=GGML_TYPE_F32||qi->type!=GGML_TYPE_I32||ai->type!=GGML_TYPE_I32||mask->type!=GGML_TYPE_I32||qm->type!=GGML_TYPE_I32||ge->type!=GGML_TYPE_F32||ow->type!=GGML_TYPE_F32)throw std::runtime_error("invalid input dtype");
    if(qi->ne[0]<1||qi->ne[1]!=1||!ggml_are_same_shape(qi,qm)||!ggml_are_same_shape(ai,mask)||ge->ne[0]!=384||ge->ne[1]!=1024||ge->ne[2]!=views||ow->ne[0]!=1024||ow->ne[1]!=views)throw std::runtime_error("invalid input dimensions");
    for(auto*t:{qi,ai})for(int64_t i=0;i<ggml_nelements(t);i++)if(((int32_t*)t->data)[i]<0||((int32_t*)t->data)[i]>=250037)throw std::runtime_error("token ID outside vocabulary");
    for(int64_t i=0;i<ggml_nelements(qm);i++)if(((int32_t*)qm->data)[i]!=1)throw std::runtime_error("question must not contain padding");
    for(int row=0;row<k;row++){bool padding=false;for(int t=0;t<mask->ne[0];t++){int v=((int32_t*)mask->data)[row*mask->ne[0]+t];if(v==0)padding=true;else if(v!=1||padding)throw std::runtime_error("candidate masks must be right padded");}}
    for(auto*t:{pixel,ge,ow})for(int64_t i=0;i<ggml_nelements(t);i++)if(!std::isfinite(((float*)t->data)[i]))throw std::runtime_error("nonfinite image input");
    for(int vi=0;vi<views;vi++){float sum=0;for(int i=0;i<1024;i++){float v=((float*)ow->data)[vi*1024+i];if(v<0||v>1)throw std::runtime_error("invalid ownership");sum+=v;}if(sum<=0)throw std::runtime_error("empty owned region");}
    auto*qraw=(int32_t*)qi->data;V q=bert(w,std::vector<int32_t>(qraw,qraw+qi->ne[0]));V qcls=mean(q);dump(debug,"q",q);
    auto*geom=(float*)in.get("geometry")->data;auto*owned=(float*)in.get("owned_weights")->data;V memory,summary(384,0);
    for(int vi=0;vi<views;vi++){
        V v=vision(w,pixel,vi);dump(debug,"vision"+std::to_string(vi),v);
        if(!vi){float denom=std::accumulate(owned,owned+1024,0.f);for(int t=0;t<1024;t++)for(int d=0;d<384;d++)summary[d]+=v[t*384+d]*owned[t]/denom;}
        V z=resample(w,v,qcls,geom+vi*1024*384,owned+vi*1024,vi?32:64);dump(debug,"z"+std::to_string(vi),z);memory.insert(memory.end(),z.begin(),z.end());
    }
    auto*types=(float*)w.get("memory_type")->data;for(size_t i=0;i<memory.size();i++)memory[i]+=types[i%384];for(size_t i=0;i<q.size();i++)memory.push_back(q[i]+types[384+i%384]);V hs;
    for(int i=0;i<k;i++){auto*ids=(int32_t*)ai->data+i*ai->ne[0];auto*valid=(int32_t*)mask->data+i*mask->ne[0];std::vector<int32_t> tokens;for(int t=0;t<ai->ne[0];t++)if(valid[t])tokens.push_back(ids[t]);if(tokens.empty())throw std::runtime_error("empty candidate");
        auto a=bert(w,tokens);auto h=fuse_choice(w,a,memory);hs.insert(hs.end(),h.begin(),h.end());}
    dump(debug,"hs",hs);V logits=score(w,hs,k);float u=answerability(w,qcls,summary,hs,k),conf=1/(1+std::exp(-u)),maximum=*std::max_element(logits.begin(),logits.end());V probs;float denom=0;for(float l:logits){probs.push_back(std::exp(l-maximum));denom+=probs.back();}for(auto &p:probs)p=p/denom*conf;
    std::ofstream out(output_path);out.precision(10);out<<"{\"option_logits\":";array(out,logits);out<<",\"answerability_logit\":"<<u<<",\"probabilities\":";array(out,probs);out<<",\"none_probability\":"<<1-conf<<",\"local_tiles\":"<<views-1<<"}\n";if(!out)throw std::runtime_error("failed output write");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
