//! Hand-written XLM-RoBERTa cross-encoder reranker forward in burn (bge-reranker-v2-m3, en-ru).
//!
//! The transformer BLOCK (self-attention, FFN, LayerNorm, encoder stack) is byte-for-byte identical
//! between BERT and XLM-RoBERTa, so this module REUSES the `Bert*` structs from [`super::model`]
//! verbatim — see [`BertEncoder`]/[`BertLayer`]. RoBERTa differs only in three places, and this file
//! carries exactly those:
//!   1. position ids: `create_position_ids_from_input_ids` (`pad + cumsum(mask) * mask`), NOT
//!      `arange(0..s)`. Non-pad tokens get incremental positions from `pad_token_id + 1`; pad tokens
//!      collapse to `pad_token_id`. For XLM-R `pad_token_id == 1`.
//!   2. `layer_norm_eps == 1e-5` (XLM-R), NOT BERT's `1e-12`; read from config.
//!   3. token-type table has ONE row (`type_vocab == 1`); the segment tensor is all-zero.
//!
//! The classification head is RoBERTa's `dense -> tanh -> out_proj` over the `<s>` (token-0) hidden
//! state — there is NO BERT-style pooler — and the output is `[batch, num_labels]` (num_labels == 1).

use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::tanh;

use super::model::{
    BertAttention, BertEncoder, BertIntermediate, BertLayer, BertOutput, BertSelfAttention,
    BertSelfOutput,
};

/// Config for the XLM-RoBERTa reranker. All fields are positional (no derive defaults) so the load
/// task (Task 4) fills them straight from the model's `config.json`. Real dims for
/// bge-reranker-v2-m3 en-ru: vocab 61696, hidden 1024, 24 layers, 16 heads, intermediate 4096,
/// max_pos 8194, pad_token_id 1, ln_eps 1e-5, num_labels 1.
///
/// The `allow` is for burn's `Config` derive, not for this file: with every field positional the
/// derive emits a `Self { vocab_size: vocab_size, … }` constructor, and clippy ≥ 1.99 lints inside
/// macro expansions and reports it against these declarations. The NLI config next door escapes
/// only because its fields carry `#[config(default = …)]` and get a builder instead.
#[allow(clippy::redundant_field_names)]
#[derive(Config, Debug)]
pub struct RobertaRerankerConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub max_pos: usize,
    pub pad_token_id: usize,
    pub ln_eps: f64,
    pub num_labels: usize,
}

/// RoBERTa input embeddings: word + position + (single-row) token-type, then LayerNorm. The only
/// behavioural difference from [`super::model::BertEmbeddings`] is the position-id derivation.
#[derive(Module, Debug)]
pub struct RobertaEmbeddings<B: Backend> {
    pub word_embeddings: Embedding<B>,
    pub position_embeddings: Embedding<B>,
    pub token_type_embeddings: Embedding<B>,
    pub layer_norm: LayerNorm<B>,
    pad_token_id: i64,
}

impl<B: Backend> RobertaEmbeddings<B> {
    fn forward(&self, ids: Tensor<B, 2, Int>, mask: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, s] = ids.dims();
        let device = ids.device();
        // RoBERTa `create_position_ids_from_input_ids`: incremental positions over non-pad tokens,
        // offset by pad_token_id. `position_ids = pad_token_id + cumsum(mask, dim=1) * mask`.
        // (mask is 1 for real tokens, 0 for pad; cumsum runs along the sequence axis.)
        let maskf = mask.float();
        let incremental = maskf.clone().cumsum(1) * maskf;
        let pos_ids = (incremental + self.pad_token_id as f64).int();
        // token_type table has a single row (type_vocab == 1): the whole segment tensor is 0.
        let types = Tensor::<B, 2, Int>::zeros([b, s], &device);
        let e = self.word_embeddings.forward(ids)
            + self.position_embeddings.forward(pos_ids)
            + self.token_type_embeddings.forward(types);
        self.layer_norm.forward(e)
    }
}

/// RoBERTa sequence-classification head: `out_proj(tanh(dense(cls)))` over the token-0 (`<s>`)
/// hidden state. Replaces the BERT pooler + linear classifier with a single two-layer head.
#[derive(Module, Debug)]
pub struct RobertaClassificationHead<B: Backend> {
    pub dense: Linear<B>,
    pub out_proj: Linear<B>,
}

impl<B: Backend> RobertaClassificationHead<B> {
    fn forward(&self, cls: Tensor<B, 2>) -> Tensor<B, 2> {
        self.out_proj.forward(tanh(self.dense.forward(cls)))
    }
}

/// The full reranker: RoBERTa embeddings -> reused BERT encoder stack -> RoBERTa head. Produces raw
/// relevance logits `[batch, num_labels]` (num_labels == 1).
#[derive(Module, Debug)]
pub struct RobertaRerankerModel<B: Backend> {
    pub embeddings: RobertaEmbeddings<B>,
    pub encoder: BertEncoder<B>,
    pub classifier: RobertaClassificationHead<B>,
}

impl RobertaRerankerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RobertaRerankerModel<B> {
        let ln = || {
            LayerNormConfig::new(self.hidden)
                .with_epsilon(self.ln_eps)
                .init(device)
        };
        let lin = |i: usize, o: usize| LinearConfig::new(i, o).init::<B>(device);
        // Reused encoder block — mirrors `BertNliConfig::init`'s layer builder verbatim.
        let layer = || BertLayer {
            attention: BertAttention {
                self_attention: BertSelfAttention {
                    query: lin(self.hidden, self.hidden),
                    key: lin(self.hidden, self.hidden),
                    value: lin(self.hidden, self.hidden),
                    heads: self.heads,
                },
                output: BertSelfOutput {
                    dense: lin(self.hidden, self.hidden),
                    layer_norm: ln(),
                },
            },
            intermediate: BertIntermediate {
                dense: lin(self.hidden, self.intermediate),
            },
            output: BertOutput {
                dense: lin(self.intermediate, self.hidden),
                layer_norm: ln(),
            },
        };
        RobertaRerankerModel {
            embeddings: RobertaEmbeddings {
                word_embeddings: EmbeddingConfig::new(self.vocab_size, self.hidden).init(device),
                position_embeddings: EmbeddingConfig::new(self.max_pos, self.hidden).init(device),
                // Single-row token-type table (type_vocab == 1).
                token_type_embeddings: EmbeddingConfig::new(1, self.hidden).init(device),
                layer_norm: ln(),
                pad_token_id: self.pad_token_id as i64,
            },
            encoder: BertEncoder {
                layer: (0..self.layers).map(|_| layer()).collect(),
            },
            classifier: RobertaClassificationHead {
                dense: lin(self.hidden, self.hidden),
                out_proj: lin(self.hidden, self.num_labels),
            },
        }
    }
}

impl<B: Backend> RobertaRerankerModel<B> {
    /// Raw relevance logits `[batch, num_labels]`. The token-type segment is all-zero internally.
    pub fn forward(&self, ids: Tensor<B, 2, Int>, mask: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let [b, s] = ids.dims();
        // Additive attention mask: (1 - mask) * -1e4 -> [b,1,1,s] (identical to `BertNliModel`).
        let add_mask = (mask.clone().float().neg() + 1.0).reshape([b, 1, 1, s]) * -1.0e4;
        let mut h = self.embeddings.forward(ids, mask);
        for layer in &self.encoder.layer {
            h = layer.forward(h, add_mask.clone());
        }
        let hidden = h.dims()[2];
        let cls = h.slice([0..b, 0..1]).reshape([b, hidden]); // token 0 (`<s>`)
        self.classifier.forward(cls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn forward_shape_is_b_by_1() {
        type B = NdArray;
        let dev = Default::default();
        // tiny config so the test is fast (2 layers), not the real 24.
        let cfg = RobertaRerankerConfig::new(
            64,   // vocab
            32,   // hidden
            2,    // layers
            4,    // heads
            64,   // interm
            40,   // max_pos
            1,    // pad
            1e-5, // ln_eps
            1,    // labels
        );
        let m = cfg.init::<B>(&dev);
        let ids = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![5i64, 6, 7, 1, 5i64, 8, 1, 1], [2, 4]),
            &dev,
        );
        let mask = burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![1i64, 1, 1, 0, 1i64, 1, 0, 0], [2, 4]),
            &dev,
        );
        let out = m.forward(ids, mask);
        assert_eq!(out.dims(), [2, 1]);
    }
}
