import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rlgames.games import GAMES, make
from rlgames.utils.constants import CHANCE


def random_playout(game, key, max_steps=200):
    """Play one game with uniform chance and uniformly random legal actions."""
    step = jax.jit(game.step)
    observe = jax.jit(game.observe)
    state = game.init()
    for _ in range(max_steps):
        if bool(state.terminal):
            return state
        key, sub = jax.random.split(key)
        if int(state.player) == CHANCE:
            probs = state.chance
        else:
            assert observe(state, state.player).shape == (game.obs_dim,)
            probs = state.legal / state.legal.sum()
        action = jax.random.choice(sub, game.num_actions, p=probs)
        state = step(state, action)
    raise AssertionError(f"{game.name} did not terminate in {max_steps} steps")


class GameTest(unittest.TestCase):
    def test_random_playouts_are_zero_sum(self):
        for name in GAMES:
            game = make(name)
            for seed in range(3):
                state = random_playout(game, jax.random.key(seed))
                self.assertAlmostEqual(float(state.returns.sum()), 0.0, msg=name)

    def test_max_game_length(self):
        from rlgames.core.tree import enumerate_tree

        for name in GAMES:
            if name == "connect_four":
                continue
            game = make(name)
            tree = enumerate_tree(game)
            self.assertLessEqual(tree.max_depth - 1, game.max_game_length, msg=name)

    def test_vmap_step(self):
        for name in GAMES:
            game = make(name)
            states = jax.tree.map(lambda x: jnp.stack([x] * 8), game.init())
            actions = jnp.argmax(states.legal | (states.chance > 0), axis=1)
            next_states = jax.jit(jax.vmap(game.step))(states, actions)
            self.assertEqual(next_states.player.shape, (8,), msg=name)

    def test_connect_four_vertical_win(self):
        game = make("connect_four")
        state = game.init()
        for action in [0, 1, 0, 1, 0, 1, 0]:
            state = game.step(state, jnp.int32(action))
        self.assertTrue(bool(state.terminal))
        np.testing.assert_allclose(state.returns, [1.0, -1.0])

    def test_connect_four_full_column_is_illegal(self):
        game = make("connect_four")
        state = game.init()
        for action in [3, 3, 3, 3, 3, 3]:
            state = game.step(state, jnp.int32(action))
        self.assertFalse(bool(state.legal[3]))
        self.assertTrue(bool(state.legal[2]))

    def test_leduc_pair_beats_high_card(self):
        game = make("leduc_poker")
        state = game.init()
        # Player 0 gets a jack, player 1 a king, and the public card is the other jack.
        for action in [0, 4]:
            state = game.step(state, jnp.int32(action))
        for action in [1, 1]:  # check, check
            state = game.step(state, jnp.int32(action))
        state = game.step(state, jnp.int32(1))
        for action in [1, 1]:
            state = game.step(state, jnp.int32(action))
        self.assertTrue(bool(state.terminal))
        np.testing.assert_allclose(state.returns, [1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
