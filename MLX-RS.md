## Use below for mlx binding in Rust

https://github.com/oxideai/mlx-rs

<div align="center">
<h1><b>mlx-rs</b></h1>

Rust bindings for Apple's mlx machine learning library.

[![Discord](https://img.shields.io/discord/1176807732473495552.svg?color=7289da&&logo=discord)](https://discord.gg/jZvTsxDX49)
[![Current Crates.io Version](https://img.shields.io/crates/v/mlx-rs.svg)](https://crates.io/crates/mlx-rs)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)]()
[![Test Status](https://github.com/oxideai/mlx-rs/actions/workflows/validate.yml/badge.svg)](https://github.com/oxideai/mlx-rs/actions/workflows/validate.yml)
[![Blaze](https://runblaze.dev/gh/307493885959233117281096297203102330146/badge.svg)](https://runblaze.dev)
[![Rust Version](https://img.shields.io/badge/Rust-1.81.0+-blue)](https://releases.rs/docs/1.81.0)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

> **⚠️ Project is in active development - contributors welcome!**

---

<div align="left" valign="middle">
<a href="https://runblaze.dev">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://www.runblaze.dev/logo_dark.png">
   <img align="right" src="https://www.runblaze.dev/logo_light.png" height="102px"/>
 </picture>
</a>

<br style="display: none;"/>

_[Blaze](https://runblaze.dev) supports this project by providing ultra-fast Apple Silicon macOS Github Action Runners. Apply the discount code `AI25` at checkout to enjoy 25% off your first year._

</div>

</div>

## Documentation

Due to known limitation of docsrs, we are hosting the documentation on github pages [here](https://oxideai.github.io/mlx-rs/mlx_rs/).

## Features

MLX is an array framework for machine learning on Apple Silicon. mlx-rs provides Rust bindings for MLX, allowing you to use MLX in your Rust projects.

Some key features of MLX and `mlx-rs` include:
- **Performance**: MLX is optimized for Apple Silicon, providing fast performance for machine learning tasks.
- **Lazy Evaluation**: MLX uses lazy evaluation to optimize performance and memory usage. Arrays are only materialized when needed.
- **Dynamic Graphs**: Computation graphs in MLX are constructed dynamically, allowing for flexible and efficient computation. Changing the shapes of function arguments does not require recompilation.
- **Mutli-Device Support**: MLX supports running computations on any of the supported devices (for now the CPU and GPU).
- **Unified memory**: MLX provides a unified memory model, meaning arrays live in the same memory space, regardless of the device they are computed on. Operations can be performed on arrays on different devices without copying data between them.

`mlx-rs` is designed to be a safe and idiomatic Rust interface to MLX, providing a seamless experience for Rust developers.

## Examples
The [examples](examples/) directory contains sample projects demonstrating different uses cases of our library.
- [mnist](examples/mnist/): Train a basic neural network on the MNIST digit dataset
- [mistral](examples/mistral/): Text generation using the pre-trained Mistral model

## Installation

Add this to your `Cargo.toml`:
```toml
[dependencies]
mlx-rs = "0.21.0"
```

## Feature Flags

* `metal` - enables metal (GPU) usage in MLX
* `accelerate` - enables using the accelerate framework in MLX

## Important Notes on Automatic Differentiation

When using automatic differentiation in mlx-rs, there's an important difference in how closures work compared to Python's MLX. In Python, variables are implicitly captured and properly traced in the compute graph. However, in Rust, we need to be more explicit about which arrays should be traced.

❌ This approach may cause segfaults:
```rust
// Don't do this
let x = random::normal::<f32>(&[num_examples, num_features], None, None, None)?;
let y = x.matmul(&w_star)? + eps;

let loss_fn = |w: &Array| -> Result<Array, Exception> {
    let y_pred = x.matmul(w)?;  // x and y are captured from outer scope
    let loss = Array::from_f32(0.5) * ops::mean(&ops::square(&(y_pred - &y))?, None, None)?;
    Ok(loss)
};

let grad_fn = transforms::grad(loss_fn, &[0]);
```

✅ Instead, pass all required arrays as inputs to ensure proper tracing:
```rust
let loss_fn = |inputs: &[Array]| -> Result<Array, Exception> {
    let w = &inputs[0];
    let x = &inputs[1];
    let y = &inputs[2];

    let y_pred = x.matmul(w)?;
    let loss = Array::from_f32(0.5) * ops::mean(&ops::square(y_pred - y)?, None, None)?;
    Ok(loss)
};
let argnums = &[0];  // Specify which argument to differentiate with respect to

// Pass all required arrays in the inputs slice
let mut inputs = vec![w, x, y];
let grad = transforms::grad(loss_fn, argnums)(&inputs)?;
```

When using gradients in training loops, remember to update the appropriate array in your inputs:

```rust
let mut inputs = vec![w, x, y];

for _ in 0..num_iterations {
    let grad = transforms::grad(loss_fn, argnums)(&inputs)?;
    inputs[0] = &inputs[0] - Array::from_f32(learning_rate) * grad;  // Update the weight array
    inputs[0].eval()?;
}
```

We are actively working on improving this API to make it more ergonomic and closer to Python's behavior. For now, explicitly passing all required arrays as shown above is the recommended approach.

## Versioning

For simplicity, the main crate `mls-rs` follows MLX’s versioning, allowing you to easily see which MLX version you’re using under the hood. The `mlx-sys` crate follows the versioning of `mlx-c`, as that is the version from which the API is generated.

## Community

If you are excited about the project or want to contribute, don't hesitate to join our [Discord](https://discord.gg/jZvTsxDX49)!
We try to be as welcoming as possible to everybody from any background. We're still building this out, but you can ask your questions there!

## Status

mlx-rs is currently in active development and can be used to run MLX models in Rust.

## MSRV

The minimum supported Rust version is 1.81.0.

The MSRV is the minimum Rust version that can be used to compile each crate.

## License

mlx-rs is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details. Opening a pull
request is assumed to signal agreement with these licensing terms.


## Example
```rust
use hf_hub::{
    api::sync::{Api, ApiBuilder, ApiRepo},
    Repo,
};
use mlx_rs::{
    array,
    module::{Module, ModuleParametersExt},
    ops::indexing::{argmax_axis, IndexOp, NewAxis},
    random::categorical,
    transforms::eval,
    Array,
};
use tokenizers::Tokenizer;

mod model;

use model::{Mistral, MistralInput, MistralOutput, ModelArgs};

type Error = Box<dyn std::error::Error + Send + Sync>;
type Result<T, E = Error> = std::result::Result<T, E>;

use clap::Parser;

#[derive(Parser)]
#[command(about = "Mistral inference example")]
pub struct Cli {
    /// The message to be processed by the model
    #[clap(long, default_value = "In the begging the Unverse was created.")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[clap(long, default_value = "100")]
    max_tokens: usize,

    /// The sampling temperature
    #[clap(long, default_value = "0.0")]
    temp: f32,

    /// The batch size of tokens to generate
    #[clap(long, default_value = "10")]
    tokens_per_eval: usize,

    /// The PRNG seed
    #[clap(long, default_value = "0")]
    seed: u64,
}

fn build_hf_api() -> Result<Api> {
    let cache_dir = std::env::var("HF_CACHE_DIR").ok();

    let mut builder = ApiBuilder::new();
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir.into());
    }
    builder.build().map_err(Into::into)
}

fn get_tokenizer(repo: &ApiRepo) -> Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let t = Tokenizer::from_file(tokenizer_filename)?;

    Ok(t)
}

fn get_model_args(repo: &ApiRepo) -> Result<ModelArgs> {
    let model_args_filename = repo.get("params.json")?;
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    Ok(model_args)
}

fn load_model(repo: &ApiRepo) -> Result<Mistral> {
    let model_args = get_model_args(repo)?;
    let mut model = Mistral::new(&model_args)?;
    let weights_filename = repo.get("weights.safetensors")?;
    model.load_safetensors(weights_filename)?;

    Ok(model)
}

fn sample(logits: &Array, temp: f32) -> Result<Array> {
    match temp {
        0.0 => argmax_axis(logits, -1, None).map_err(Into::into),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical(logits, None, None, None).map_err(Into::into)
        }
    }
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

struct Generate<'a> {
    model: &'a mut Mistral,
    temp: f32,
    state: GenerateState<'a>,
}

enum GenerateState<'a> {
    Start {
        prompt_token: &'a Array,
    },
    Continue {
        y: Array,
        cache: Vec<Option<(Array, Array)>>,
    },
}

impl<'a> Generate<'a> {
    pub fn new(model: &'a mut Mistral, prompt_token: &'a Array, temp: f32) -> Self {
        Self {
            model,
            temp,
            state: GenerateState::Start { prompt_token },
        }
    }
}

impl Iterator for Generate<'_> {
    type Item = Result<Array>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Start { prompt_token } => {
                let initial_cache = Vec::with_capacity(0); // This won't allocate
                let input = MistralInput {
                    inputs: prompt_token,
                    cache: &initial_cache,
                };
                let MistralOutput { logits, cache } = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));

                self.state = GenerateState::Continue {
                    y: y.clone(),
                    cache,
                };

                Some(Ok(y))
            }
            GenerateState::Continue { y, cache } => {
                let next_token = y.index((.., NewAxis));
                let input = MistralInput {
                    inputs: &next_token,
                    cache: cache.as_slice(),
                };
                let MistralOutput {
                    logits,
                    cache: new_cache,
                } = tri!(self.model.forward(input));

                let logits = tri!(logits.squeeze_axes(&[1]));
                let y = tri!(sample(&logits, self.temp));

                self.state = GenerateState::Continue {
                    y: y.clone(),
                    cache: new_cache,
                };

                Some(Ok(y))
            }
        }
    }
}

fn main() -> Result<()> {
    // If you want to manually set the cache directory, you can set the HF_CACHE_DIR
    // environment variable or put it in a .env file located at the root of this example
    // (ie. examples/mistral/.env)
    let _ = dotenv::dotenv();
    let api = build_hf_api()?;

    // Parse args
    let cli = Cli::parse();

    mlx_rs::random::seed(cli.seed)?;

    // The model used in the original example is converted to safetensors and
    // uploaded to the huggingface hub
    let model_id = "minghuaw/Mistral-7B-v0.1".to_string();
    let repo = api.repo(Repo::new(model_id, hf_hub::RepoType::Model));
    println!("[INFO] Loading model... ");
    let tokenizer = get_tokenizer(&repo)?;
    let mut model = load_model(&repo)?;

    model = mlx_rs::nn::quantize(model, None, None)?;

    let encoding = tokenizer.encode(&cli.prompt[..], true)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    print!("{}", cli.prompt);

    let generate = Generate::new(&mut model, &prompt_tokens, cli.temp);
    let mut tokens = Vec::with_capacity(cli.max_tokens);
    for (token, ntoks) in generate.zip(0..cli.max_tokens) {
        let token = token?;
        tokens.push(token);

        if ntoks == 0 {
            eval(&tokens)?;
        }

        if tokens.len() % cli.tokens_per_eval == 0 {
            eval(&tokens)?;
            let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
            let s = tokenizer.decode(&slice, true)?;
            print!("{s}");
        }
    }

    eval(&tokens)?;
    let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
    let s = tokenizer.decode(&slice, true)?;
    println!("{s}");

    println!("------");

    Ok(())
}

use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::concatenate_axis,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub dim: i32,
    pub n_layers: i32,
    pub head_dim: i32,
    pub hidden_dim: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub norm_eps: f32,
    pub vocab_size: i32,
    pub rope_theta: Option<f32>,
}

impl ModelArgs {
    pub const DEFAULT_ROPE_THETA: f32 = 10000.0;
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    n_heads: i32,
    n_kv_heads: i32,
    repeats: i32,
    scale: f32,

    #[quantizable]
    #[param]
    wq: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    wk: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    wv: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    wo: MaybeQuantized<nn::Linear>,

    #[param]
    rope: nn::Rope,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let n_heads = args.n_heads;
        let n_kv_heads = args.n_kv_heads;
        let repeats = n_heads / n_kv_heads;
        let scale = (args.head_dim as f32).powf(-0.5);

        let wq = nn::LinearBuilder::new(args.dim, n_heads * args.head_dim)
            .bias(false)
            .build()?;
        let wk = nn::LinearBuilder::new(args.dim, n_kv_heads * args.head_dim)
            .bias(false)
            .build()?;
        let wv = nn::LinearBuilder::new(args.dim, n_kv_heads * args.head_dim)
            .bias(false)
            .build()?;
        let wo = nn::LinearBuilder::new(n_heads * args.head_dim, args.dim)
            .bias(false)
            .build()?;
        let rope = nn::RopeBuilder::new(args.head_dim)
            .traditional(true)
            .base(args.rope_theta.unwrap_or(ModelArgs::DEFAULT_ROPE_THETA))
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            repeats,
            scale,
            wq: MaybeQuantized::new(wq),
            wk: MaybeQuantized::new(wk),
            wv: MaybeQuantized::new(wv),
            wo: MaybeQuantized::new(wo),
            rope,
        })
    }
}

struct AttentionInput<'a> {
    x: &'a Array,
    mask: Option<ScaledDotProductAttentionMask<'a>>,
    cache: Option<(&'a Array, &'a Array)>,
}

struct AttentionOutput {
    output: Array,
    cache: (Array, Array),
}

impl Module<AttentionInput<'_>> for Attention {
    type Output = AttentionOutput;

    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        // NOTE: this will panic if the input shape is not correct
        let B = x.shape()[0];
        let L = x.shape()[1];

        let mut queries = self.wq.forward(x)?;
        let mut keys = self.wk.forward(x)?;
        let mut values = self.wv.forward(x)?;

        // Prepare the queries, keys, and values for the attention computation
        queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        match cache {
            Some((key_cache, value_cache)) => {
                let offset = key_cache.shape()[2];
                queries = self.rope.forward((&queries, offset))?;
                keys = self.rope.forward((&keys, offset))?;
                keys = concatenate_axis(&[key_cache, &keys], 2)?;
                values = concatenate_axis(&[value_cache, &values], 2)?;
            }
            None => {
                queries = self.rope.forward(&queries)?;
                keys = self.rope.forward(&keys)?;
            }
        }

        let output = scaled_dot_product_attention(queries, &keys, &values, self.scale, mask)?;
        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        let output = self.wo.forward(&output)?;

        Ok(AttentionOutput {
            output,
            cache: (keys, values),
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.wq.training_mode(mode);
        self.wk.training_mode(mode);
        self.wv.training_mode(mode);
        self.wo.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct FeedForward {
    #[quantizable]
    #[param]
    w1: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    w2: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    w3: MaybeQuantized<nn::Linear>,
}

impl FeedForward {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let w1 = nn::LinearBuilder::new(args.dim, args.hidden_dim)
            .bias(false)
            .build()?;
        let w2 = nn::LinearBuilder::new(args.hidden_dim, args.dim)
            .bias(false)
            .build()?;
        let w3 = nn::LinearBuilder::new(args.dim, args.dim)
            .bias(false)
            .build()?;
        Ok(Self {
            w1: MaybeQuantized::new(w1),
            w2: MaybeQuantized::new(w2),
            w3: MaybeQuantized::new(w3),
        })
    }
}

impl Module<&Array> for FeedForward {
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, x: &'_ Array) -> Result<Self::Output, Self::Error> {
        let w2_input = nn::silu(self.w1.forward(x)?)?.multiply(self.w3.forward(x)?)?;
        self.w2.forward(&w2_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.w1.training_mode(mode);
        self.w2.training_mode(mode);
        self.w3.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct TransformerBlock {
    n_heads: i32,
    dim: i32,

    #[quantizable]
    #[param]
    attention: Attention,

    #[quantizable]
    #[param]
    feed_forward: FeedForward,

    #[param]
    attention_norm: nn::RmsNorm,

    #[param]
    ffn_norm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let n_heads = args.n_heads;
        let dim = args.dim;

        let attention = Attention::new(args)?;
        let feed_forward = FeedForward::new(args)?;
        let attention_norm = nn::RmsNormBuilder::new(dim).eps(args.norm_eps).build()?;
        let ffn_norm = nn::RmsNormBuilder::new(dim).eps(args.norm_eps).build()?;
        Ok(Self {
            n_heads,
            dim,
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }
}

impl Module<AttentionInput<'_>> for TransformerBlock {
    type Output = AttentionOutput;

    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;
        let norm_x = self.attention_norm.forward(x)?;
        let attention_input = AttentionInput {
            x: &norm_x,
            mask,
            cache,
        };
        let attention_output = self.attention.forward(attention_input)?;

        let r = attention_output.output;
        let cache = attention_output.cache;

        let h = x.add(r)?;
        let r = self.feed_forward.forward(&self.ffn_norm.forward(&h)?)?;
        let output = h.add(r)?;

        Ok(AttentionOutput { output, cache })
    }

    fn training_mode(&mut self, mode: bool) {
        self.attention.training_mode(mode);
        self.feed_forward.training_mode(mode);
        self.attention_norm.training_mode(mode);
        self.ffn_norm.training_mode(mode);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MistralError {
    #[error("Invalid vocab size: {0}")]
    InvalidVocabSize(i32),

    #[error(transparent)]
    Exception(#[from] Exception),
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mistral {
    vocab_size: i32,
    n_layers: i32,

    #[quantizable]
    #[param]
    tok_embeddings: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    layers: Vec<TransformerBlock>,

    #[param]
    norm: nn::RmsNorm,

    #[quantizable]
    #[param]
    output: MaybeQuantized<nn::Linear>,
}

impl Mistral {
    pub fn new(args: &ModelArgs) -> Result<Self, MistralError> {
        let vocab_size = args.vocab_size;
        if vocab_size <= 0 {
            // We would still have to check for the zero case even if we switch to u32
            return Err(MistralError::InvalidVocabSize(vocab_size));
        }
        let n_layers = args.n_layers;

        let tok_embeddings = nn::Embedding::new(vocab_size, args.dim)?;
        let layers = (0..n_layers)
            .map(|_| TransformerBlock::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.dim)
            .eps(args.norm_eps)
            .build()?;
        let output = nn::LinearBuilder::new(args.dim, vocab_size)
            .bias(false)
            .build()?;

        Ok(Self {
            vocab_size,
            n_layers,
            tok_embeddings: MaybeQuantized::new(tok_embeddings),
            layers,
            norm,
            output: MaybeQuantized::new(output),
        })
    }
}

pub struct MistralInput<'a> {
    pub inputs: &'a Array,
    pub cache: &'a [Option<(Array, Array)>],
}
pub struct MistralOutput {
    pub logits: Array,
    pub cache: Vec<Option<(Array, Array)>>,
}

impl Module<MistralInput<'_>> for Mistral {
    type Output = MistralOutput;

    type Error = MistralError;

    fn forward(&mut self, input: MistralInput<'_>) -> Result<Self::Output, Self::Error> {
        let MistralInput { inputs, cache } = input;

        let mut h = self.tok_embeddings.forward(inputs)?;

        let mut mask = None;
        if h.shape()[1] > 1 {
            let mask_ = nn::MultiHeadAttention::create_additive_causal_mask::<f32>(h.shape()[1])?;
            let mask_ = mask_.as_dtype(h.dtype())?;
            mask = Some(mask_);
        }

        let mut out_cache = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let cache_entry = cache.get(i).and_then(Option::as_ref).map(|(k, v)| (k, v));
            let input = AttentionInput {
                x: &h,
                mask: mask.as_ref().map(Into::into),
                cache: cache_entry,
            };
            let output = layer.forward(input)?;
            h = output.output;
            out_cache.push(Some(output.cache));
        }

        let output = self.output.forward(&self.norm.forward(&h)?)?;

        Ok(MistralOutput {
            logits: output,
            cache: out_cache,
        })
    }

    fn training_mode(&mut self, mode: bool) {
        self.tok_embeddings.training_mode(mode);
        self.layers
            .iter_mut()
            .for_each(|layer| layer.training_mode(mode));
        self.norm.training_mode(mode);
        self.output.training_mode(mode);
    }
}

use mlx_rs::error::Exception;
use mlx_rs::{ops, transforms, Array};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let num_features: i32 = 100;
    let num_examples: i32 = 1000;
    let num_iterations: i32 = 10000;
    let learning_rate: f32 = 0.01;

    // True weight vector
    // let w_star = mlx_rs::random::normal::<f32>(&[num_features], None, None, None)?;
    let w_star = mlx_rs::normal!(shape = &[num_features])?;

    // Input examples (design matrix)
    // let x = mlx_rs::random::normal::<f32>(&[num_examples, num_features], None, None, None)?;
    let x = mlx_rs::normal!(shape = &[num_examples, num_features])?;

    // Noisy labels
    // let eps = mlx_rs::random::normal::<f32>(&[num_examples], None, None, None)? * 1e-2;
    let eps = mlx_rs::normal!(shape = &[num_examples])? * 1e-2;
    let y = x.matmul(&w_star)? + eps;

    // Initialize random weights
    // let w = mlx_rs::random::normal::<f32>(&[num_features], None, None, None)? * 1e-2;
    let w = mlx_rs::normal!(shape = &[num_features])? * 1e-2;

    let loss_fn = |inputs: &[Array]| -> Result<Array, Exception> {
        let w = &inputs[0];
        let x = &inputs[1];
        let y = &inputs[2];

        let y_pred = x.matmul(w)?;
        let loss = Array::from_f32(0.5) * ops::mean(&ops::square(y_pred - y)?, None)?;
        Ok(loss)
    };

    let mut grad_fn = transforms::grad(loss_fn);

    let now = std::time::Instant::now();
    let mut inputs = [w, x, y];

    for _ in 0..num_iterations {
        let grad = grad_fn(&inputs)?;
        inputs[0] = &inputs[0] - Array::from_f32(learning_rate) * grad;
        inputs[0].eval()?;
    }

    let elapsed = now.elapsed();

    let loss = loss_fn(&inputs)?;
    let error_norm = ops::sum(&ops::square(&(&inputs[0] - &w_star))?, None)?.sqrt()?;
    let throughput = num_iterations as f32 / elapsed.as_secs_f32();

    println!(
        "Loss {:.5}, L2 distance: |w-w*| = {:.5}, Throughput {:.5} (it/s)",
        loss.item::<f32>(),
        error_norm.item::<f32>(),
        throughput
    );

    Ok(())
}

use mlx_rs::transforms::grad;
use mlx_rs::{Array, Dtype};

fn scalar_basics() {
    // create a scalar array
    let x: Array = 1.0.into();

    // the datatype is .float32
    let dtype = x.dtype();
    assert_eq!(dtype, Dtype::Float32);

    // get the value
    let s = x.item::<f32>();
    assert_eq!(s, 1.0);

    // reading the value with a different type is a fatal error
    // let i = x.item::<i32>();

    // scalars have a size of 1
    let size = x.size();
    assert_eq!(size, 1);

    // scalars have 0 dimensions
    let ndim = x.ndim();
    assert_eq!(ndim, 0);

    // scalar shapes are empty arrays
    let shape = x.shape();
    assert!(shape.is_empty());
}

#[allow(unused_variables)]
fn array_basics() {
    // make a multidimensional array.
    let x: Array = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // mlx is row-major by default so the first row of this array
    // is [1.0, 2.0] and the second row is [3.0, 4.0]

    // Make an array of shape {2, 2} filled with ones:
    let y = Array::ones::<f32>(&[2, 2]).unwrap();

    // Pointwise add x and y:
    let z = x.add(&y);

    // Same thing:
    let mut z = &x + &y;

    // mlx is lazy by default. At this point `z` only
    // has a shape and a type but no actual data:
    assert_eq!(z.dtype(), Dtype::Float32);
    assert_eq!(z.shape(), vec![2, 2]);

    // To actually run the computation you must evaluate `z`.
    // Under the hood, mlx records operations in a graph.
    // The variable `z` is a node in the graph which points to its operation
    // and inputs. When `eval` is called on an array (or arrays), the array and
    // all of its dependencies are recursively evaluated to produce the result.
    // Once an array is evaluated, it has data and is detached from its inputs.
    z.eval().unwrap();

    // Of course the array can still be an input to other operations. You can even
    // call eval on the array again, this will just be a no-op:
    z.eval().unwrap(); // no-op

    // Some functions or methods on arrays implicitly evaluate them. For example
    // accessing a value in an array or printing the array implicitly evaluate it:
    z = Array::ones::<f32>(&[1]).unwrap();
    z.item::<f32>(); // implicit evaluation

    z = Array::ones::<f32>(&[2, 2]).unwrap();
    println!("{z}"); // implicit evaluation
}

fn automatic_differentiation() {
    use mlx_rs::error::Result;

    fn f(x: &Array) -> Result<Array> {
        x.square()
    }

    fn calculate_grad(func: impl Fn(&Array) -> Result<Array>, arg: &Array) -> Result<Array> {
        grad(&func)(arg)
    }

    let x = Array::from(1.5);

    let dfdx = calculate_grad(f, &x).unwrap();
    assert_eq!(dfdx.item::<f32>(), 2.0 * 1.5);

    let dfdx2 = calculate_grad(|args| calculate_grad(f, args), &x).unwrap();
    assert_eq!(dfdx2.item::<f32>(), 2.0);
}

fn main() {
    scalar_basics();
    array_basics();
    automatic_differentiation();
}
```