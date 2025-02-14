import streamlit as st
import torch
import numpy as np
import os
import types
from typing import Optional, List
import gc

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'init_state' not in st.session_state:
    st.session_state.init_state = None
if 'init_out' not in st.session_state:
    st.session_state.init_out = None

def initialize_model(device: str, float_mode: str) -> None:
    """Initialize the RWKV model with given parameters."""
    args = types.SimpleNamespace()

    # Model configuration
    args.RUN_DEVICE = device
    args.FLOAT_MODE = float_mode
    args.MODEL_NAME = 'SpikeGPT-216M'
    args.n_layer = 18
    args.n_embd = 768
    args.ctx_len = 1024
    args.vocab_size = 50277
    args.head_qk = 0
    args.pre_ffn = 0
    args.grad_cp = 0
    args.my_pos_emb = 0

    os.environ["RWKV_JIT_ON"] = '1'
    os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

    from src.model_run import RWKV_RNN
    st.session_state.model = RWKV_RNN(args)

    # Initialize tokenizer
    from src.utils import TOKENIZER
    st.session_state.tokenizer = TOKENIZER(
        ["20B_tokenizer.json", "20B_tokenizer.json"],
        UNKNOWN_CHAR=None
    )

def generate_text(
    prompt: str,
    max_length: int,
    temperature: float,
    top_p: float,
    top_p_newline: float
) -> str:
    """Generate text based on the given prompt and parameters."""
    if not st.session_state.tokenizer.charMode:
        ctx = st.session_state.tokenizer.tokenizer.encode(prompt)
    else:
        prompt = st.session_state.tokenizer.refine_context(prompt)
        ctx = [st.session_state.tokenizer.stoi.get(s, st.session_state.tokenizer.UNKNOWN_CHAR)
               for s in prompt]

    src_len = len(ctx)
    src_ctx = ctx.copy()

    # Initialize states
    init_state = None
    mem1 = None
    mem2 = None

    # Process initial prompt
    for i in range(src_len):
        x = ctx[: i + 1]
        if i == src_len - 1:
            init_out, init_state, mem1, mem2 = st.session_state.model.forward(
                x, init_state, mem1, mem2
            )
        else:
            init_state, mem1, mem2 = st.session_state.model.forward(
                x, init_state, mem1, mem2, preprocess_only=True
            )

    # Generate new text
    out_last = src_len
    generated_text = ""

    for i in range(src_len, src_len + max_length):
        x = ctx[: i + 1]
        x = x[-st.session_state.model.args.ctx_len:]

        if i == src_len:
            out = init_out.clone()
            state = init_state.clone()
        else:
            out, state, mem1, mem2 = st.session_state.model.forward(x, state, mem1, mem2)

        if not st.session_state.tokenizer.charMode:
            out[0] = -999999999  # disable <|endoftext|>

        token = st.session_state.tokenizer.sample_logits(
            out,
            x,
            st.session_state.model.args.ctx_len,
            temperature=temperature,
            top_p_usual=top_p,
            top_p_newline=top_p_newline,
        )
        token = int(token)
        ctx += [token]

        if st.session_state.tokenizer.charMode:
            char = st.session_state.tokenizer.itos[token]
            generated_text += char
        else:
            char = st.session_state.tokenizer.tokenizer.decode(ctx[out_last:])
            if '\ufffd' not in char:  # is valid utf8 string?
                generated_text += char
                out_last = i + 1

    return generated_text

# Streamlit UI
st.title("SpikeGPT")

# Sidebar for model configuration
st.sidebar.header("Model Configuration")
device = st.sidebar.selectbox("Device", ["cpu", "cuda"],
                            help="Select the device to run the model on")
float_mode = st.sidebar.selectbox("Float Mode",
                                ["fp32", "fp16", "bf16"],
                                help="Select floating point precision")

# Initialize model button
if st.sidebar.button("Initialize Model"):
    with st.spinner("Initializing model... This may take a few minutes."):
        initialize_model(device, float_mode)
    st.success("Model initialized successfully!")

# Main interface for text generation
st.header("Text Generation")

# Input parameters
prompt = st.text_area("Enter your prompt:", height=150)
max_length = st.slider("Maximum length to generate:",
                      min_value=10, max_value=1000, value=333)
temperature = st.slider("Temperature:",
                       min_value=0.1, max_value=2.0, value=1.5)
top_p = st.slider("Top P:", min_value=0.1, max_value=1.0, value=0.7)
top_p_newline = st.slider("Top P Newline:",
                         min_value=0.1, max_value=1.0, value=0.9)

# Generate button
if st.button("Generate Text"):
    if st.session_state.model is None:
        st.error("Please initialize the model first using the sidebar!")
    else:
        try:
            with st.spinner("Generating text..."):
                generated_text = generate_text(
                    prompt, max_length, temperature, top_p, top_p_newline
                )
                st.text_area("Generated Text:", generated_text, height=300)

            # Clear GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            st.error(f"An error occurred during text generation: {str(e)}")

# Display model status
if st.session_state.model is not None:
    st.sidebar.success("Model is loaded and ready!")
else:
    st.sidebar.warning("Model is not initialized")

# Add some helpful information
with st.expander("How to use this interface"):
    st.markdown("""
    1. First, initialize the model using the sidebar controls
    2. Enter your prompt in the text area
    3. Adjust the generation parameters:
        - Temperature: Higher values make the output more random
        - Top P: Controls the cumulative probability threshold for sampling
        - Top P Newline: Specific threshold for newline characters
        - Maximum length: How many tokens to generate
    4. Click "Generate Text" to create the output

    Note: The first generation might take longer as the model needs to process the initial prompt.
    """)
