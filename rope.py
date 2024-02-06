from jax import (
    Array, 
    numpy as jnp, 
    lax
)

class PositionalEmbedding:
    """```
    Sinusoidal Fixed Positional Embeddings
    Args:
        maxlen:int
        dim:int
    sinusoidal_embeddings: 
        pos_emb: (1, maxlen, dim)
    get_freqs:
        get_freqs: sin_freqs(1, maxlen, 1, dim), cos_freqs(1, maxlen, 1, dim)
    ```"""
    def __init__(self, maxlen:int, dim:int):
        p, i = jnp.meshgrid(jnp.arange(float(maxlen)), jnp.arange(dim/2)*2)
        theta = (p/1e4**(i/dim)).T

        self.pos_emb = jnp.stack([jnp.sin(theta), jnp.cos(theta)], axis=-1)
        self.pos_emb = self.pos_emb.reshape((maxlen, dim))[None] # (1, maxlen, dim)

    def sinusoidal_embeddings(self):
        return self.pos_emb # (1, maxlen, dim)
    
    def get_freqs(self):
        sin_freqs = jnp.repeat(self.pos_emb[..., None, ::2], repeats=2, axis=-1)
        cos_freqs = jnp.repeat(self.pos_emb[..., None, 1::2], repeats=2, axis=-1)
        return sin_freqs, cos_freqs # (1, maxlen, 1, dim), (1, maxlen, 1, dim)
    

def apply_rotary_embeddings(q:Array, k:Array, sin_freqs:Array, cos_freqs:Array):
    T = q.shape[1]

    minus_swap_alternate = lambda x: jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)

    q = q*cos_freqs[:, :T, :, :] + minus_swap_alternate(q)*sin_freqs[:, :T, :, :] # (B, T, h, dq)*(1, T, 1, dq) + (B, T, h, dq)*(1, T, 1, dq)
    k = k*cos_freqs[:, :T, :, :] + minus_swap_alternate(k)*sin_freqs[:, :T, :, :] # (B, T, h, dq)*(1, T, 1, dq) + (B, T, h, dq)*(1, T, 1, dq)
    return q, k # (B, T, h, dq), (B, T, h, dq)



########################################## Also Check ##########################################

def precompute_freqs(dim: int, maxlen: int, theta: float = 1e4):
    freqs = 1.0 / (theta ** (jnp.arange(0., float(dim), 2.)[: (dim // 2)] / dim))
    t = jnp.arange(maxlen)
    freqs = jnp.outer(t, freqs)
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    return freqs_sin, freqs_cos  # (maxlen, dim/2), (maxlen, dim/2)

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(shape)

def apply_rotary_emb(
    xq:Array,
    xk:Array,
    freqs_sin:Array,
    freqs_cos:Array
) -> tuple[Array, Array]:
    
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = jnp.moveaxis(xq.reshape(xq.shape[:-1] + (-1, 2)), -1, 0)
    xk_r, xk_i = jnp.moveaxis(xk.reshape(xk.shape[:-1] + (-1, 2)), -1, 0)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = lax.collapse(jnp.stack([xq_out_r, xq_out_i], axis=-1), 3)
    xk_out = lax.collapse(jnp.stack([xk_out_r, xk_out_i], axis=-1), 3)

    return xq_out, xk_out