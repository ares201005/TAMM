import numpy as np
import pytamm
import time


def main() -> None:

    with pytamm.TammContext() as ctx:

        ni, nj, nk, nm = 500, 200, 400, 100
        tile = 50
        nsteps = 5

        a = ctx.zeros((ni, nj, nk), tile=tile)  # (ni, nj, nk)
        b = ctx.zeros((nk, nm), tile=tile)      # (nk, nm)

        # In MPI runs, all ranks must load identical NumPy data before
        # from_numpy; otherwise different ranks race to write different values.
        rng = np.random.default_rng(12345)
        a_np = rng.random((ni, nj, nk))
        b_np = rng.random((nk, nm))

        ctx.from_numpy(a, a_np)
        ctx.from_numpy(b, b_np)
        c = ctx.einsum("ijk,km->ijm", a, b)
        t0 = time.time()
        for _ in range(nsteps):
            # c(i,j,m) = a(i,j,k) * b(k,m)
            ctx.einsum("ijk,km->ijm", a, b, out=c)
        wt1 = time.time() - t0

        t0 = time.time()
        for _ in range(nsteps):
            c_ref = np.einsum("ijk,km->ijm", a_np, b_np)
        wt = time.time() - t0

        c_tamm = ctx.to_numpy(c)
        err = np.linalg.norm(c_tamm - c_ref)
        c_norm = ctx.norm(c)
        if ctx.rank() == 0:
            print("wall time (numpy) is:", wt)
            print("wall time (tamm)  is:", wt1)
            print("norm(C_tamm)       = ", c_norm)
            print("||C_tamm - C_ref|| = ", err)

        ctx.deallocate(a, b, c)

if __name__ == "__main__":
    main()
