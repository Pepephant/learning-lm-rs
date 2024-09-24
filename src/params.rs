use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

use std::fs::File;
use memmap2::MmapOptions;

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...    
        // };
        
        let filename = "models/story/model.safetensors";
        let file = File::open(filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();

        let get_tensor: Box<dyn Fn(&str) -> Tensor<f32>> = Box::new(|name: &str| {
            let tensor = tensors.tensor(name).unwrap();
            
            // 将 tensor 数据转换为 fp32 形式
            let data: Vec<f32> = tensor.data().chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            
            let result = Tensor::new(data, &tensor.shape().to_vec());
            result
        });


        LLamaParams { 
            embedding_table: get_tensor("lm_head.weight") , 
            rms_att_w: vec![get_tensor("model.layers.0.input_layernorm.weight"), get_tensor("model.layers.1.input_layernorm.weight")],
            wq: vec![get_tensor("model.layers.0.self_attn.q_proj.weight"), get_tensor("model.layers.1.self_attn.q_proj.weight")], 
            wk: vec![get_tensor("model.layers.0.self_attn.k_proj.weight"), get_tensor("model.layers.1.self_attn.k_proj.weight")], 
            wv: vec![get_tensor("model.layers.0.self_attn.v_proj.weight"), get_tensor("model.layers.1.self_attn.v_proj.weight")], 
            wo: vec![get_tensor("model.layers.0.self_attn.o_proj.weight"), get_tensor("model.layers.1.self_attn.o_proj.weight")], 
            rms_ffn_w: vec![get_tensor("model.layers.0.post_attention_layernorm.weight"), get_tensor("model.layers.1.post_attention_layernorm.weight")],
            w_up: vec![get_tensor("model.layers.0.mlp.up_proj.weight"), get_tensor("model.layers.1.mlp.up_proj.weight")], 
            w_gate: vec![get_tensor("model.layers.0.mlp.gate_proj.weight"), get_tensor("model.layers.1.mlp.gate_proj.weight")], 
            w_down: vec![get_tensor("model.layers.0.mlp.down_proj.weight"), get_tensor("model.layers.1.mlp.down_proj.weight")], 
            rms_out_w: get_tensor("model.norm.weight") , 
            lm_head: get_tensor("lm_head.weight") 
        }
        
        // LLamaParams {
        //     embedding_table: get_tensor(...),
        //     ...
        // }
    }

    pub fn test_safetensors() -> Self {
        let filename = "models/story/model.safetensors";
        let file = File::open(filename).unwrap();
        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer).unwrap();

        let get_tensor: Box<dyn Fn(&str) -> Tensor<f32>> = Box::new(|name: &str| {
            let tensor = tensors.tensor(name).unwrap();
            
            // 将 tensor 数据转换为 fp32 形式
            let data: Vec<f32> = tensor.data().chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            
            let result = Tensor::new(data, &tensor.shape().to_vec());
            result
        });


        LLamaParams { 
            embedding_table: get_tensor("lm_head.weight") , 
            rms_att_w: vec![get_tensor("model.layers.0.input_layernorm.weight"), get_tensor("model.layers.1.input_layernorm.weight")],
            wq: vec![get_tensor("model.layers.0.self_attn.q_proj.weight"), get_tensor("model.layers.1.self_attn.q_proj.weight")], 
            wk: vec![get_tensor("model.layers.0.self_attn.k_proj.weight"), get_tensor("model.layers.1.self_attn.k_proj.weight")], 
            wv: vec![get_tensor("model.layers.0.self_attn.v_proj.weight"), get_tensor("model.layers.1.self_attn.v_proj.weight")], 
            wo: vec![get_tensor("model.layers.0.self_attn.o_proj.weight"), get_tensor("model.layers.1.self_attn.o_proj.weight")], 
            rms_ffn_w: vec![get_tensor("model.layers.0.post_attention_layernorm.weight"), get_tensor("model.layers.1.post_attention_layernorm.weight")],
            w_up: vec![get_tensor("model.layers.0.mlp.up_proj.weight"), get_tensor("model.layers.1.mlp.up_proj.weight")], 
            w_gate: vec![get_tensor("model.layers.0.mlp.gate_proj.weight"), get_tensor("model.layers.1.mlp.gate_proj.weight")], 
            w_down: vec![get_tensor("model.layers.0.mlp.down_proj.weight"), get_tensor("model.layers.1.mlp.down_proj.weight")], 
            rms_out_w: get_tensor("model.norm.weight") , 
            lm_head: get_tensor("lm_head.weight") 
        }
        
    }
}

#[test]
pub fn test_loads() {
    let llama = LLamaParams::test_safetensors();
    
    assert!(float_close(llama.embedding_table.data()[50], 0.14453125, 1e-6));
    assert_eq!(llama.lm_head.data()[10], llama.embedding_table.data()[10]);
    assert!(float_close(llama.rms_att_w[0].data()[10], 0.18652344, 1e-6));
    assert!(float_close(llama.rms_ffn_w[1].data()[10], 0.32421875, 1e-6));
    assert!(float_close(llama.rms_out_w.data()[100], 0.73046875, 1e-6));
    assert!(float_close(llama.w_down[0].data()[100], -0.0625, 1e-6));
    assert!(float_close(llama.w_up[0].data()[100], 1.46875, 1e-6));
    assert!(float_close(llama.w_gate[1].data()[100], 0.296875, 1e-6));
    assert!(float_close(llama.wq[1].data()[100], 0.032226563, 1e-6));
    assert!(float_close(llama.wk[1].data()[100], -0.21386719, 1e-6));
    assert!(float_close(llama.wv[0].data()[100], 0.041015625, 1e-6));
    assert!(float_close(llama.wo[0].data()[100], 0.01965332, 1e-6));
}

pub fn float_close(x: f32, y: f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}