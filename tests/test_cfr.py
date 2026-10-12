"""Checks rlgames/core/cfr.py (yours) against rlgames/reference/cfr.py.

Every test runs twice: `ReferenceCFRTest` runs it on the reference, which shows the
test itself is right, and `YourCFRTest` runs it on your implementation, skipping
anything still raising NotImplementedError.

    uv run pytest tests/test_cfr.py -v
"""

import functools
import unittest

import numpy as np

from rlgames.core import cfr as yours
from rlgames.core.exploitability import exploitability
from rlgames.core.tree import enumerate_tree, expected_values, node_policy, uniform_policy
from rlgames.games import make
from rlgames.reference import cfr as reference


@functools.cache
def tree_of(name, **kwargs):
    return enumerate_tree(make(name, **kwargs))


def skip_if_unimplemented(test):
    @functools.wraps(test)
    def wrapper(self):
        try:
            test(self)
        except NotImplementedError as e:
            self.skipTest(str(e))

    return wrapper


class CFRChecks:
    impl = None  # module under test, set by the subclasses below

    def assert_policies_match(self, tree, num_iters, avg_power=1.0, check_every=1):
        """Runs both implementations side by side and compares their current and
        average policies after every `check_every` iterations."""
        cfg = dict(num_iters=num_iters, avg_power=avg_power)
        mine = self.impl.CFRPlus(tree, self.impl.CFRConfig(**cfg))
        ref = reference.CFRPlus(tree, reference.CFRConfig(**cfg))
        for t in range(1, num_iters + 1):
            mine.iterate()
            ref.iterate()
            if t % check_every:
                continue
            for name in ("current_policy", "average_policy"):
                np.testing.assert_allclose(
                    getattr(mine, name)(), getattr(ref, name)(), atol=1e-9,
                    err_msg=f"{name} differs from the reference after iteration {t}",
                )

    # regret matching

    @skip_if_unimplemented
    def test_regret_matching_examples(self):
        legal = np.array([[True, True, True], [True, True, False], [True, False, True]])
        regrets = np.array([[1.0, 3.0, -2.0], [-1.0, -5.0, 0.0], [0.0, 7.0, 0.0]])
        expected = np.array([
            [0.25, 0.75, 0.0],  # proportional to positive regret
            [0.5, 0.5, 0.0],  # nothing positive: uniform over legal actions
            [0.5, 0.0, 0.5],  # regret on an illegal action is ignored
        ])
        np.testing.assert_allclose(self.impl.regret_matching(regrets, legal), expected)

    @skip_if_unimplemented
    def test_regret_matching_random(self):
        rng = np.random.default_rng(0)
        legal = rng.random((200, 5)) < 0.6
        legal[np.arange(200), rng.integers(0, 5, 200)] = True
        regrets = rng.normal(size=(200, 5)) - 0.5
        regrets[:20] = -1.0  # some rows with no positive regret
        np.testing.assert_allclose(
            self.impl.regret_matching(regrets, legal),
            reference.regret_matching(regrets, legal.astype(np.float64)),
            atol=1e-12,
        )

    # CFR+ iterates, compared to the reference

    @skip_if_unimplemented
    def test_starts_uniform(self):
        tree = tree_of("kuhn_poker")
        solver = self.impl.CFRPlus(tree, self.impl.CFRConfig())
        np.testing.assert_allclose(solver.current_policy(), uniform_policy(tree))
        np.testing.assert_allclose(solver.average_policy(), uniform_policy(tree))

    @skip_if_unimplemented
    def test_matches_reference_rock_paper_scissors(self):
        self.assert_policies_match(tree_of("biased_rock_paper_scissors"), 50)

    @skip_if_unimplemented
    def test_matches_reference_kuhn(self):
        self.assert_policies_match(tree_of("kuhn_poker"), 50)

    @skip_if_unimplemented
    def test_matches_reference_leduc(self):
        self.assert_policies_match(tree_of("leduc_poker"), 20, check_every=5)

    @skip_if_unimplemented
    def test_matches_reference_goofspiel(self):
        # Simultaneous moves: player 1 must not see player 0's bid.
        self.assert_policies_match(tree_of("goofspiel", num_cards=3), 10)

    @skip_if_unimplemented
    def test_matches_reference_quadratic_averaging(self):
        self.assert_policies_match(tree_of("kuhn_poker"), 20, avg_power=2.0)

    @skip_if_unimplemented
    def test_matches_reference_tic_tac_toe(self):
        # Many boards are reached by several move orders (transpositions).
        self.assert_policies_match(tree_of("tic_tac_toe"), 3)

    # convergence

    @skip_if_unimplemented
    def test_solves_kuhn(self):
        tree = tree_of("kuhn_poker")
        policy = self.impl.CFRPlus(tree, self.impl.CFRConfig(num_iters=1000)).solve()
        self.assertLess(exploitability(tree, policy), 1e-3)
        # Player 0's value of Kuhn poker is -1/18.
        value = expected_values(tree, node_policy(tree, policy))[0]
        self.assertAlmostEqual(value, -1 / 18, places=3)

    @skip_if_unimplemented
    def test_solves_biased_rock_paper_scissors(self):
        tree = tree_of("biased_rock_paper_scissors")
        policy = self.impl.CFRPlus(tree, self.impl.CFRConfig(num_iters=3000)).solve()
        np.testing.assert_allclose(policy, [[0.25, 0.5, 0.25]] * 2, atol=1e-2)


class ReferenceCFRTest(CFRChecks, unittest.TestCase):
    impl = reference


class YourCFRTest(CFRChecks, unittest.TestCase):
    impl = yours


if __name__ == "__main__":
    unittest.main()
