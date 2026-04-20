import argparse
import os

import numpy as np
import pytamm


def main() -> None:

    with pytamm.Context(
        distribution=pytamm.DistributionKind.NW,
        memory_manager=pytamm.MemoryManagerKind.GA,
    ) as ctx:

        ni, nj, nk, nm = 200, 100, 80, 40
        tile = 20

        i = ctx.tiled_index_space(ni, tile)
        j = ctx.tiled_index_space(nj, tile)
        k = ctx.tiled_index_space(nk, tile)
        m = ctx.tiled_index_space(nm, tile)

        a = ctx.tensor([i, j, k])  # (ni, nj, nk)
        b = ctx.tensor([k, m])     # (nk, nm)
        c = ctx.tensor([i, j, m])  # (ni, nj, nm)

        ctx.allocate(a, b, c)

        a_np = np.random.rand(ni, nj, nk)
        b_np = np.random.rand(nk, nm)

        ctx.from_numpy(a, a_np)
        ctx.from_numpy(b, b_np)
        ctx.fill(c, 0.0)

        # c(i,j,m) = a(i,j,k) * b(k,m)
        ctx.contract(
            c,
            ["i", "j", "m"],
            a,
            ["i", "j", "k"],
            b,
            ["k", "m"],
        )

        c_tamm = ctx.to_numpy(c)
        c_ref = np.einsum("ijk,km->ijm", a_np, b_np)

        err = np.linalg.norm(c_tamm - c_ref)
        # if ctx.rank() == 0:
        print("||C_tamm - C_ref|| =", err)
        print("norm(C_tamm) =", ctx.norm(c))

        ctx.deallocate(a, b, c)

if __name__ == "__main__":
    main()
