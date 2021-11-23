"""Resource management without queue information."""
from typing import Optional, Any
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class MicroService:
    """Micro service."""

    def __init__(self, init_servers: int = 0):
        self.queue = []
        self.n_servers = init_servers  # number of servers
        self.n_active = 0  # number of active servers

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


def single_service(params: dict[str, Any]):  # pylint: disable=too-many-statements
    """Simulates a single micro service.

    Args:
        params = {
            "max_dep": 300000, # max number of departures in the simulation
            "arr_rate": 0.8, # job arrival rate
            "deadline": 23.1, # response time deadline
            "qos_prob": 0.99, # QoS requirement: P(resp. time <= deadline) >= qos_prob
            "upper": 51, # upper threshold for deficit
            "lower": -51, # lower threshold for deficit
        }
    """
    micro = MicroService(1)
    deficit = 0
    time = 0
    rng = np.random.default_rng()
    response_times = []
    n_server_list = []
    deficit_list = []
    queue_list = []
    while len(response_times) < params["max_dep"]:
        next_arr_time = rng.exponential(1 / params["arr_rate"])
        next_dep_time = 0
        if micro.n_active:
            next_dep_time = rng.exponential(1 / micro.n_active)
        # next event is departure
        if next_dep_time and next_dep_time < next_arr_time:
            # id of the departing job
            idx = rng.integers(micro.n_active)
            response = time + next_dep_time - micro.complete(idx)["arr_time"]
            response_times.append(response)
            if response <= params["deadline"]:
                deficit -= 1
            # threshold policy
            if deficit < params["lower"]:
                micro.reduce_server()
            elif deficit > params["upper"]:
                micro.serve()
                micro.add_server()
                micro.serve()
            else:
                micro.serve()
            time += next_dep_time
        # next event is arrival
        else:
            # observe system state
            n_server_list.append(micro.n_servers)
            deficit_list.append(deficit)
            queue_list.append(len(micro.queue))
            # dynamics
            micro.arrive(time + next_arr_time)
            if rng.random() < params["qos_prob"]:
                deficit += 1
            # try to see if the job can directly go into service
            micro.serve()
            # threshold policy
            if deficit < params["lower"]:
                micro.reduce_server()
            elif deficit > params["upper"]:
                micro.add_server()
                micro.serve()
            time += next_arr_time
    print(np.mean(response_times[-int(params["max_dep"] / 4) :]))
    print(np.mean(queue_list[-int(params["max_dep"] / 4) :]))
    print(np.mean(n_server_list[-int(params["max_dep"] / 4) :]))
    plt.plot(n_server_list[0:200000], label="# of servers")
    plt.plot(deficit_list[0:200000], label="deficit")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(queue_list[0:200000])
    plt.show()
    plt.hist(response_times, 30)
    plt.show()
