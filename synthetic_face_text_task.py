"""Synthetic sanity task for facial expression fusion with text.

This script trains a minimal classifier that maps facial expression features to
text labels. It is meant as a quick check that the AU/embedding path is wired
correctly before running expensive experiments. Optionally a ``.npz`` dataset
with ``x`` (features) and ``y`` (integer labels) can be provided.
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def generate_dataset(num_samples: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create a toy dataset of facial features and text labels."""
    # Four discrete expressions represented as one-hot AUs.
    base = jnp.eye(4)
    # Repeat each expression roughly num_samples / 4 times.
    reps = num_samples // 4
    feats = jnp.tile(base, (reps, 1))
    labels = jnp.arange(4).repeat(reps)
    return feats, labels


def load_np_dataset(path: Path) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load dataset from ``.npz`` containing ``x`` and ``y`` arrays."""
    data = np.load(path)
    return jnp.asarray(data["x"]), jnp.asarray(data["y"])


@dataclass
class LinearModel:
    w: jnp.ndarray
    b: jnp.ndarray

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.w + self.b


def init_model(key: jax.Array) -> LinearModel:
    w_key, b_key = jax.random.split(key)
    w = jax.random.normal(w_key, (4, 4)) * 0.01
    b = jnp.zeros((4,))
    return LinearModel(w, b)


def loss_fn(model: LinearModel, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = model(x)
    loss = jnp.mean(jax.nn.softmax_cross_entropy_with_integer_labels(logits, y))
    return loss


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(model: LinearModel, _unused, batch) -> Tuple[LinearModel, any]:
    """Single gradient descent step."""
    x, y = batch
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    new_model = LinearModel(
        w=model.w - 0.1 * grads.w,
        b=model.b - 0.1 * grads.b,
    )
    return new_model, None, loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to .npz file with 'x' and 'y' arrays (facial features and labels)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="training epochs")
    parser.add_argument("--samples", type=int, default=100, help="synthetic sample count")
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)

    if args.dataset:
        x, y = load_np_dataset(args.dataset)
    else:
        x, y = generate_dataset(args.samples)

    model = init_model(key)

    for step in range(args.epochs):
        model, _, loss = train_step(model, None, (x, y))
        if step % (args.epochs // 4 or 1) == 0:
            print(f"step {step} loss {loss:.4f}")

    preds = jnp.argmax(model(x), axis=-1)
    acc = jnp.mean((preds == y).astype(jnp.float32))
    print("final accuracy", float(acc))


if __name__ == "__main__":
    main()
