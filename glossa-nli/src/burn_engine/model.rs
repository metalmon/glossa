//! Hand-written RuBERT-base three-way NLI forward in burn (Plan 4). The module tree mirrors the HF
//! `BertForSequenceClassification` naming so weights load with a minimal key remap
//! (`.self.`->`.self_attention.`, `LayerNorm`->`layer_norm`). Validated CPU+GPU parity to ORT
//! (~1e-6 CPU, ~3e-6 Vulkan/SPIR-V f32).

use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{gelu, softmax, tanh};

#[derive(Config, Debug)]
pub struct BertNliConfig {
    #[config(default = 119547)]
    pub vocab_size: usize,
    #[config(default = 768)]
    pub hidden: usize,
    #[config(default = 12)]
    pub layers: usize,
    #[config(default = 12)]
    pub heads: usize,
    #[config(default = 3072)]
    pub intermediate: usize,
    #[config(default = 512)]
    pub max_pos: usize,
    #[config(default = 2)]
    pub type_vocab: usize,
    #[config(default = 3)]
    pub num_labels: usize,
    #[config(default = 1e-12)]
    pub ln_eps: f64,
}

#[derive(Module, Debug)]
pub struct BertEmbeddings<B: Backend> {
    pub word_embeddings: Embedding<B>,
    pub position_embeddings: Embedding<B>,
    pub token_type_embeddings: Embedding<B>,
    pub layer_norm: LayerNorm<B>,
}

impl<B: Backend> BertEmbeddings<B> {
    fn forward(&self, ids: Tensor<B, 2, Int>, types: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_b, s] = ids.dims();
        let device = ids.device();
        let pos_ids = Tensor::<B, 1, Int>::arange(0..s as i64, &device).reshape([1, s]);
        let e = self.word_embeddings.forward(ids)
            + self.position_embeddings.forward(pos_ids) // broadcasts over batch
            + self.token_type_embeddings.forward(types);
        self.layer_norm.forward(e)
    }
}

#[derive(Module, Debug)]
pub struct BertSelfAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    heads: usize,
}

impl<B: Backend> BertSelfAttention<B> {
    fn forward(&self, h: Tensor<B, 3>, add_mask: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, s, hidden] = h.dims();
        let hd = hidden / self.heads;
        let shape = [b, s, self.heads, hd];
        let q = self.query.forward(h.clone()).reshape(shape).swap_dims(1, 2);
        let k = self.key.forward(h.clone()).reshape(shape).swap_dims(1, 2);
        let v = self.value.forward(h).reshape(shape).swap_dims(1, 2);
        let scale = (hd as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale + add_mask;
        let probs = softmax(scores, 3);
        let ctx = probs.matmul(v); // [b,H,s,hd]
        ctx.swap_dims(1, 2).reshape([b, s, hidden])
    }
}

#[derive(Module, Debug)]
pub struct BertSelfOutput<B: Backend> {
    pub dense: Linear<B>,
    pub layer_norm: LayerNorm<B>,
}
impl<B: Backend> BertSelfOutput<B> {
    fn forward(&self, hidden: Tensor<B, 3>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.layer_norm.forward(self.dense.forward(hidden) + input)
    }
}

#[derive(Module, Debug)]
pub struct BertAttention<B: Backend> {
    pub self_attention: BertSelfAttention<B>,
    pub output: BertSelfOutput<B>,
}
impl<B: Backend> BertAttention<B> {
    fn forward(&self, h: Tensor<B, 3>, add_mask: Tensor<B, 4>) -> Tensor<B, 3> {
        let attn = self.self_attention.forward(h.clone(), add_mask);
        self.output.forward(attn, h)
    }
}

#[derive(Module, Debug)]
pub struct BertIntermediate<B: Backend> {
    pub dense: Linear<B>,
}
#[derive(Module, Debug)]
pub struct BertOutput<B: Backend> {
    pub dense: Linear<B>,
    pub layer_norm: LayerNorm<B>,
}

#[derive(Module, Debug)]
pub struct BertLayer<B: Backend> {
    pub attention: BertAttention<B>,
    pub intermediate: BertIntermediate<B>,
    pub output: BertOutput<B>,
}
impl<B: Backend> BertLayer<B> {
    fn forward(&self, h: Tensor<B, 3>, add_mask: Tensor<B, 4>) -> Tensor<B, 3> {
        let a = self.attention.forward(h, add_mask);
        let inter = gelu(self.intermediate.dense.forward(a.clone())); // erf-exact GELU
        self.output.layer_norm.forward(self.output.dense.forward(inter) + a)
    }
}

#[derive(Module, Debug)]
pub struct BertEncoder<B: Backend> {
    pub layer: Vec<BertLayer<B>>,
}

#[derive(Module, Debug)]
pub struct BertPooler<B: Backend> {
    pub dense: Linear<B>,
}

#[derive(Module, Debug)]
pub struct BertModel<B: Backend> {
    pub embeddings: BertEmbeddings<B>,
    pub encoder: BertEncoder<B>,
    pub pooler: BertPooler<B>,
}

#[derive(Module, Debug)]
pub struct BertNliModel<B: Backend> {
    pub bert: BertModel<B>,
    pub classifier: Linear<B>,
}

impl BertNliConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertNliModel<B> {
        let ln = || LayerNormConfig::new(self.hidden).with_epsilon(self.ln_eps).init(device);
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).init::<B>(device);
        let layer = || BertLayer {
            attention: BertAttention {
                self_attention: BertSelfAttention {
                    query: lin(self.hidden, self.hidden),
                    key: lin(self.hidden, self.hidden),
                    value: lin(self.hidden, self.hidden),
                    heads: self.heads,
                },
                output: BertSelfOutput { dense: lin(self.hidden, self.hidden), layer_norm: ln() },
            },
            intermediate: BertIntermediate { dense: lin(self.hidden, self.intermediate) },
            output: BertOutput { dense: lin(self.intermediate, self.hidden), layer_norm: ln() },
        };
        BertNliModel {
            bert: BertModel {
                embeddings: BertEmbeddings {
                    word_embeddings: EmbeddingConfig::new(self.vocab_size, self.hidden).init(device),
                    position_embeddings: EmbeddingConfig::new(self.max_pos, self.hidden).init(device),
                    token_type_embeddings: EmbeddingConfig::new(self.type_vocab, self.hidden).init(device),
                    layer_norm: ln(),
                },
                encoder: BertEncoder { layer: (0..self.layers).map(|_| layer()).collect() },
                pooler: BertPooler { dense: lin(self.hidden, self.hidden) },
            },
            classifier: lin(self.hidden, self.num_labels),
        }
    }
}

impl<B: Backend> BertNliModel<B> {
    /// Raw 3-way logits `[batch, num_labels]`.
    pub fn forward(
        &self,
        ids: Tensor<B, 2, Int>,
        mask: Tensor<B, 2, Int>,
        types: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let [b, s] = ids.dims();
        // additive attention mask: (1 - mask) * -1e4 -> [b,1,1,s]
        let add_mask = (mask.float().neg() + 1.0).reshape([b, 1, 1, s]) * -1.0e4;
        let mut h = self.bert.embeddings.forward(ids, types);
        for layer in &self.bert.encoder.layer {
            h = layer.forward(h, add_mask.clone());
        }
        let hidden = self.bert.pooler.dense.weight.dims()[0];
        let cls = h.slice([0..b, 0..1]).reshape([b, hidden]); // token 0
        let pooled = tanh(self.bert.pooler.dense.forward(cls));
        self.classifier.forward(pooled)
    }
}
