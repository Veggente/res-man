"""Tests eroc."""
import unittest
import numpy as np
import resman


class TestResman(unittest.TestCase):
    """Tests resman."""

    def test_stat_dist(self):
        """Tests stationary distribution calculation."""
        self.assertAlmostEqual(sum(resman.stat_dist(30, 100, 0.8, 1)), 1, places=4)

    def test_tail(self):
        """Tests tail probability."""
        n_servers = 2
        max_queue = 20
        arrival_rate = 0.8
        service_rate = 1
        n_trials = 100000
        delay_threshold = 3
        tail = resman.response_time_tail(
            n_servers, delay_threshold, max_queue, arrival_rate, service_rate
        )
        pmf = [
            float(x)
            for x in resman.stat_dist(n_servers, max_queue, arrival_rate, service_rate)
        ]
        rng = np.random.default_rng(0)
        tail_emp = []
        for i in range(max_queue + 1):
            waiting = rng.exponential(scale=1 / service_rate, size=n_trials)
            if i >= n_servers:
                waiting += rng.gamma(
                    i - n_servers + 1, 1 / n_servers / service_rate, size=n_trials
                )
            tail_emp.append(sum(waiting > delay_threshold) / len(waiting))
        self.assertAlmostEqual(tail, np.inner(tail_emp, pmf), places=2)

    def test_opt_n(self):
        """Tests optimal number of servers."""
        arrival = 0.8
        service = 1
        deadline = 22.8
        tail = 0.01
        self.assertEqual(
            resman.optimal_n_servers(deadline, tail, 100, arrival, service), 2
        )
