# Grok-1

This repository contains JAX example code for loading and running the Grok-1 open-weights model.

Make sure to download the checkpoint and place the `ckpt-0` directory in `checkpoints` - see [Downloading the weights](#downloading-the-weights)

Then, run

```shell
pip install -r requirements.txt
python run.py
```

to test the code.

The script loads the checkpoint and samples from the model on a test input.

Due to the large size of the model (314B parameters), a machine with enough GPU memory is required to test the model with the example code.
The implementation of the MoE layer in this repository is not efficient. The implementation was chosen to avoid the need for custom kernels to validate the correctness of the model.

# Model Specifications

Grok-1 is currently designed with the following specifications:

- **Parameters:** 314B
- **Architecture:** Mixture of 8 Experts (MoE)
- **Experts Utilization:** 2 experts used per token
- **Layers:** 64
- **Attention Heads:** 48 for queries, 8 for keys/values
- **Embedding Size:** 6,144
- **Tokenization:** SentencePiece tokenizer with 131,072 tokens
- **Additional Features:**
  - Rotary embeddings (RoPE)
  - Supports activation sharding and 8-bit quantization
- **Maximum Sequence Length (context):** 8,192 tokens

# Downloading the weights

You can download the weights using a torrent client and this magnet link:

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

or directly using [HuggingFace 🤗 Hub](https://huggingface.co/xai-org/grok-1):
```
git clone https://github.com/xai-org/grok-1.git && cd grok-1
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

# License

The code and associated Grok-1 weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of Grok-1.

## Synthetic facial-expression sanity task

The script [`synthetic_face_text_task.py`](./synthetic_face_text_task.py) trains a
tiny classifier that maps facial-action units to text labels. It does **not**
use the Grok-1 model directly; instead it serves as a quick end-to-end check
that facial embeddings can flow through JAX code.

Run the test with synthetic data:

```bash
python synthetic_face_text_task.py --epochs 200
```

If you have a dataset saved as `data.npz` containing arrays `x` (AU vectors) and
`y` (integer labels), point the script at it:

```bash
python synthetic_face_text_task.py --dataset data.npz --epochs 500
```

To create `data.npz` from your own video, first extract facial-action units with
any tool such as [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). Save
the resulting feature matrix as `x` and the corresponding labels as `y` using
NumPy:

```python
np.savez("data.npz", x=feature_array, y=label_array)
```

The final accuracy printed by the script should be close to 1.0 on the toy data
and will vary on real data. This provides a lightweight sanity check before you
wire facial embeddings into larger models.
