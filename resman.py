"""Resource management without queue information."""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    single_service()
