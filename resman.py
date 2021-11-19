"""Resource management without queue information."""
from typing import Optional
from decimal import Decimal, getcontext
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import scipy.special as sc

plt.style.use("ggplot")


class MicroService:
    """Micro service."""

    def __init__(self, init_servers: int = 0):
        self.queue = []
        self.n_servers = init_servers
        self.n_active = 0

    def arrive(self, arr_time: float):
        """A task arrives."""
        self.queue.append({"arr_time": arr_time})

    def serve(self) -> int:
        """Attempts to put a task in the queue into service.

        Returns:
            If a task is put into service.
        """
        if len(self.queue) > self.n_active and self.n_active < self.n_servers:
            self.n_active += 1
            return 1
        return 0

    def complete(self, server_idx: int) -> Optional[dict[str, float]]:
        """Completes a task.

        Args:
            server_idx: Index of server where a task completes.

        Returns:
            The completed task if any.
        """
        if self.n_active > server_idx:
            self.n_active -= 1
            return self.queue.pop(server_idx)
        return None

    def add_server(self):
        """Adds an idle server."""
        self.n_servers += 1

    def reduce_server(self):
        """Shuts down an idle server if possible."""
        if self.n_servers > self.n_active:
            self.n_servers -= 1


def single_service():
    """Simulates a single micro service."""
    micro = MicroService(1)
    active = 0
    deficit = 0
    time = 0
    rng = np.random.default_rng()
    params = {
        "upper": 1e10,
        "lower": -1e10,
        "max_time": 1000000,
        "arr_rate": 0.8,
        "deadline": 10,
        "deficit_prob": 0.2,
    }
    response_times = []
    n_server_list = []
    deficit_list = []
    queue_list = []
    while time < params["max_time"]:
        next_arr_time = rng.exponential(1 / params["arr_rate"])
        next_dep_time = 0
        if active:
            next_dep_time = rng.exponential(1 / active)
        if next_dep_time and next_dep_time < next_arr_time:
            idx = rng.integers(active)
            response = time + next_dep_time - micro.complete(idx)["arr_time"]
            active -= 1
            response_times.append(response)
            if response <= params["deadline"]:
                deficit -= 1
            if deficit < params["lower"]:
                micro.reduce_server()
            else:
                active += micro.serve()
            time += next_dep_time
        else:
            n_server_list.append(micro.n_servers)
            deficit_list.append(deficit)
            queue_list.append(len(micro.queue))
            micro.arrive(time + next_arr_time)
            if rng.random() < params["deficit_prob"]:
                deficit += 1
            if deficit > params["upper"]:
                micro.add_server()
            active += micro.serve()
            time += next_arr_time
    print(np.mean(response_times[-500000:]))
    print(np.mean(queue_list[-500000:]))
    # plt.plot(n_server_list, label="# of servers")
    # plt.plot(deficit_list, label="deficit")
    # plt.legend()
    # plt.show()
    # plt.figure()
    # plt.hist(response_times, 30)
    # plt.show()
    # plt.plot(queue_list)
    # plt.show()


def optimal_n_servers(
    delay_threshold: float, tail_prob: float, max_queue: int, rho: float
) -> int:
    """Finds the optimal number of servers.

    Args:
        delay_threshold: Maximum desired delay.
        tail_prob: Maximum tail probability.
        max_queue: Maximum queue length.
        rho: System load.

    Returns:
        Optimal number of servers.
    """

    def response_short(n_servers: int) -> float:
        """Short function for response time tail.

        Args:
            n_servers: Number of servers.

        Returns:
            Tail probability.
        """
        return response_time_tail(n_servers, delay_threshold, max_queue, rho)

    # Exponential expansion.
    right = 1
    while response_short(right) > tail_prob:
        right *= 2
    left = int(right / 2)
    # Binary search.
    while left != right - 1:
        middle = int((left + right) / 2)
        if response_short(middle) > tail_prob:
            left = middle
        else:
            right = middle
    return right


def response_time_tail(
    n_servers: int, delay_threshold: float, max_queue: int, rho: float
) -> float:
    """Calculates the response time tail probability.

    Args:
        n_servers: Number of servers.
        delay_threshold: Maximum desired delay.
        max_queue: Maximum queue length.
        rho: System load.

    Returns:
        Tail probability.
    """
    service_rate = 1
    resp = [expon.sf(delay_threshold * service_rate)] * min(n_servers, max_queue + 1)
    # i is the queue length seen at arrival.
    # See waiting.pdf for the analysis.
    first = np.exp(-service_rate * delay_threshold)
    second = np.exp(-n_servers * service_rate * delay_threshold)
    second_cumu = 0
    for i in range(n_servers, max_queue + 1):
        second_cumu += second
        if n_servers > 1:
            first *= n_servers / (n_servers - 1)
            first_whole = (
                first
                # False positive by pylint.
                * sc.gammainc(  # pylint: disable=no-member
                    i - n_servers + 1, (n_servers - 1) * service_rate * delay_threshold
                )
            )
        else:
            first *= (service_rate * delay_threshold) / i
            first_whole = first
        resp.append(first_whole + second_cumu)
        second *= n_servers * service_rate * delay_threshold / (i - n_servers + 1)
    pmf = [float(x) for x in stat_dist(n_servers, max_queue, rho)]
    return np.inner(resp, pmf)


def stat_dist(n_servers: int, max_queue: int = 20, rho: float = 0.8) -> list[Decimal]:
    """Calculates stationary distribution of queue length.

    Args:
        n_servers: Number of servers.
        max_queue: Maximum queue length after truncation.  Probability
            masses of larger queue lengths are dropped and the pmf is
            not normalized.
        rho: System load.

    Returns:
        Pmf of the queue length stationary distribution.
    """
    getcontext().prec = 28
    service_rate = 1
    arrival_rate = n_servers * Decimal(rho) * Decimal(service_rate)
    pi0 = Decimal(0)
    for i in range(n_servers):
        pi0 = Decimal(pi0) + Decimal(n_servers * Decimal(rho)) ** Decimal(i) / Decimal(
            math.factorial(i)
        )
    pi0 = Decimal(pi0) + Decimal(n_servers * Decimal(rho)) ** Decimal(
        n_servers
    ) / Decimal(math.factorial(n_servers)) / Decimal(1 - rho)
    pi0 = Decimal(1) / Decimal(pi0)
    dist = [pi0]
    # i + 1 is the queue length.
    for i in range(min(n_servers, max_queue)):
        dist.append(dist[i] * arrival_rate / service_rate / (i + 1))
    # If max_queue has been reached, no more calculation is done.
    for i in range(n_servers, max_queue):
        dist.append(dist[i] * Decimal(rho))
    return dist


if __name__ == "__main__":
    print("optimal:", optimal_n_servers(2, 0.1, 100, 0.8))
