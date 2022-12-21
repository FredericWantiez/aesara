import jax.numpy as jnp

from jax import random, lax, jit
from collections import namedtuple

# Container for static/pre-computed parameters
BTRDParams = namedtuple(
    'BTRDParams',
    ['a', 'b', 'c', 'spq', 'vr', 'ur', 'm', 'logp', 'log1p', 'alpha']
)


def binv(key, n, p):
    """ Binomial Inversion Sampling

    Used when sequential search is fairly fast n * p < 10

    Parameters:
        - key: PRNGKey = RNG Key
        - n: Int = Number of trials
        - p: Float = Success probability
    """

    def cond_fun(val):
        _, _, accu = val
        return accu <= n

    def body_fun(val):
        k, key, accu = val
        lkey, rkey = random.split(key, num=2)
        u = random.uniform(lkey)
        geom = jnp.ceil(jnp.log(u)/_log1p)
        accu += geom
        return k+1, rkey, accu

    _log1p = jnp.log1p(-p)
    init_val = (-1, key, 0)
    k, _, _ = lax.while_loop(cond_fun, body_fun, init_val)
    return k


def stirling_approx(k):
    """ Stirling approximation of log(f(k))

    Parameters:
        - k: int = k-th order of the stirling approximation
    """
    table = jnp.array(
        [
            0.08106146679532726,
            0.04134069595540929,
            0.02767792568499834,
            0.02079067210376509,
            0.01664469118982119,
            0.01387612882307075,
            0.01189670994589177,
            0.01041126526197209,
            0.009255462182712733,
            0.008330563433362871,
        ]
    )
    def _fc(k):
        return (1/12 - (1/360 - 1/1260/(k+1)**2)/(k+1)**2)/(k+1)

    return jnp.where(k < 10, table[k], _fc(k))


def setup_btrd(n, p):
    """ Pre-compute static parameters for the BTRD algorithm (Page 4-5)
    """
    q = 1 - p
    mu = n * p
    spq = jnp.sqrt(n*p*q)
    b = 1.15 + 2.53*spq
    a = -0.0873 + 0.0248*b + 0.01*p
    c = mu + 0.5
    vr = 0.92 - 4.2/b
    ur = 0.43
    m = jnp.ceil((n+1) * p)
    logp = jnp.log(p)
    log1minusp = jnp.log1p(-p)
    alpha = (2.83 + 5.1/b)*spq
    params = BTRDParams(a, b, c, spq, vr, ur, m, logp, log1minusp, alpha)
    return params


def Gfunc(u, a, b, c):
    """ Dominating measure approximation
    """
    return (2*a/(0.5 - jnp.abs(u)) + b)*u + c

def Gpfunc(u, a, b, c):
    """ Dominating measure derivative
    """
    return a/(0.5 - jnp.abs(u))**2 + b

def btrd(key, n, p):
    """ Binomial Transformed Rejection Sampling

    Reference:
        The Generation of Binomial Random Variates - Hormann

    """
    params = setup_btrd(n, p)
    G = lambda u: Gfunc(u, params.a, params.b, params.c)
    Gp = lambda u: Gpfunc(u, params.a, params.b, params.c)

    def body_fun(val):
        _, key, _, _ = val

        ukey, vkey = random.split(key, num=2)
        v = random.uniform(vkey)
        k1 = jnp.floor(G(v/params.vr - params.ur))

        def left_branch(val):
            _, v, key = val
            lkey, rkey = random.split(key)
            u = random.uniform(lkey) - 0.5
            return (u, v, rkey)

        def right_branch(val):
            _, v, key = val
            lkey, rkey = random.split(key)
            u = v/params.vr - (params.ur + 0.5)
            u = jnp.sign(u) * 0.5 - u
            vn = random.uniform(lkey, minval=0, maxval=params.vr)
            return (u, vn, rkey)

        u2, v2, nkey = lax.cond(
            v >= params.vr,
            left_branch,
            right_branch,
            (0.5, v, vkey)
        )
        k2 = jnp.floor(G(u2))
        v, k = lax.cond(
            v < 2*params.ur*params.vr,
            lambda _: (v, k1),
            lambda _: (v2, k2),
            (0,0)
        )
        return k, nkey, u2, v

    def cond_fun(val):
        k, key, u, v = val
        incorrect = (k < 0) | (k > n) # Early rejection since it's invalid
        early = (jnp.abs(u) <= params.ur) & (v <= params.vr) # Early
        # Acceptance ratio
        m = params.m
        logp = params.logp
        log1mp = params.log1p
        fc = lambda k: stirling_approx(k.astype(int))
        logf = (
            (m + 0.5) * (jnp.log((m + 1)/(n - m + 1)) + log1mp - logp)
            + (n + 1) * jnp.log((n - m + 1.0)/(n - k + 1.0))
            + (k + 0.5) * (jnp.log((n - k + 1.0)/(k + 1.0)) + logp - log1mp)
            + fc(m)
            + fc(n - m)
            - fc(k)
            - fc(n - k)
        )
        cond = (jnp.log((params.alpha * v) / Gp(u)) > logf) | (incorrect | early)
        return cond

    k, _, _, _ = lax.while_loop(
        cond_fun,
        body_fun,
        (-1, key, 0, 0)
    )
    return k.astype(int)

_MU_THRESHOLD = 10

@jit
def binomial(key, n, p):
    mu = n * p
    flip = p > 0.5
    ps = jnp.where(~flip, p, 1-p)
    k = lax.cond(
        mu <= _MU_THRESHOLD,
        lambda val: binv(*val),
        lambda val: btrd(*val),
        (key, n, ps),
    )
    return jnp.where(~flip, k, n - k)


def binomial_sampling(key, n, p, size=None, dtype=None):
    shape = size or lax.broadcast_shapes(jnp.shape(n), jnp.shape(p))
    n = jnp.reshape(jnp.broadcast_to(n, shape), -1)
    p = jnp.reshape(jnp.broadcast_to(p, shape), -1)
    key = random.split(key, jnp.size(p))
    ret = lax.map(lambda x: binomial(*x), (key, n, p))
    return jnp.reshape(ret, shape)
