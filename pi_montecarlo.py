import numpy as np
from mpi4py import MPI


def estimate_pi(n: int, block: int = int(1e6)) -> int:
    """
    Samples are drawn in blocks smaller than n to avoid allocation of large arrays and out of memory errors. Samples consists of uniform random (x,y) pairs, where x ∈ [-1,1] and y ∈ [-1,1]. A running count is kept of samples falling inside the unit circle. Returns an estimate of pi.
    """

    total_samples_in_circle = 0
    i = 0

    while i < n:
        if n - i < block:
            block = n - i

        samples = 2 * np.random.random((block, 2)) - 1
        in_unit_circle = np.linalg.norm(samples, axis=-1) <= 1.0
        samples_in_circle = np.sum(in_unit_circle)
        total_samples_in_circle += samples_in_circle
        i += block

    return total_samples_in_circle


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = int(1e7)

    # Initialize random number generator with unique seed for each rank.
    seed = 42
    np.random.seed(seed + rank)
    in_unit_circle = np.zeros(1, dtype=int)

    if rank == size - 1:
        n = (N // size) + (N % size)
        in_unit_circle = estimate_pi(n)
    else:
        n = N // size
        in_unit_circle = estimate_pi(n)

    total_in_unit_circle = np.zeros(1, dtype=int)
    comm.Reduce(
        [in_unit_circle, MPI.INT], [total_in_unit_circle, MPI.INT], op=MPI.SUM, root=0
    )

    pi = ((4.0 * total_in_unit_circle) / N)[0]
    if rank == 0:
        print(f"Estimated value of pi: {pi:.7f}")
