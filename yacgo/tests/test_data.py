import unittest
import numpy as np
from yacgo.data import (
    Inference,
    State,
    TrainState,
    TrainingBatch,
    PrioritizedTrainState,
    DATA_DTYPE,
)


class TestPrioritizedTrainState(unittest.TestCase):
    """Sanity test for creating PrioritizedTrainState from TrainState"""

    def test_create_prio_train_state(self):
        """Quick test to create PrioritizedTrainState from TrainState"""
        ts = TrainState(
            state=np.random.random((17, 19, 19)).astype(DATA_DTYPE),
            value=np.float32(0.5),
            policy=np.random.random((362,)).astype(DATA_DTYPE),
        )
        pts = PrioritizedTrainState.from_train_state(ts)

        self.assertTrue(np.allclose(pts.state, ts.state))
        self.assertTrue(np.allclose(pts.policy, ts.policy))
        self.assertTrue(np.allclose(pts.value, ts.value))
        self.assertTrue(pts.state.dtype == ts.state.dtype)
        self.assertTrue(pts.policy.dtype == ts.policy.dtype)
        self.assertTrue(pts.value.dtype == ts.value.dtype)
        self.assertTrue(pts.state.shape == ts.state.shape)
        self.assertTrue(pts.policy.shape == ts.policy.shape)
        self.assertTrue(pts.value.shape == ts.value.shape)


class TestTrainingBatch(unittest.TestCase):
    """Sanity test on TrainingBatch class"""

    def test_pack_unpack_training_batch(self):
        """Test pack and unpack training batch methods"""
        bs = 8
        states = np.random.random((bs, 17, 19, 19)).astype(DATA_DTYPE)
        policies = np.random.random((bs, 362)).astype(DATA_DTYPE)
        values = np.random.random((bs,)).astype(DATA_DTYPE)

        tb = TrainingBatch(bs, states, values, policies)

        packed = tb.pack()
        tb2 = TrainingBatch.unpack(packed)

        self.assertTrue(np.allclose(tb2.states, tb.states))
        self.assertTrue(np.allclose(tb2.policies, tb.policies))
        self.assertTrue(np.allclose(tb2.values, tb.values))
        self.assertTrue(tb.batch_size == tb2.batch_size)
        self.assertTrue(tb2.states.dtype == tb.states.dtype)
        self.assertTrue(tb2.policies.dtype == tb.policies.dtype)
        self.assertTrue(tb2.values.dtype == tb.values.dtype)
        self.assertTrue(tb2.states.shape == tb.states.shape)
        self.assertTrue(tb2.policies.shape == tb.policies.shape)
        self.assertTrue(tb2.values.shape == tb.values.shape)


class TestTrainState(unittest.TestCase):
    """Sanity test on TrainState class"""

    def test_pack_unpack_train_state(self):
        """Test pack and unpack train state methods"""
        state = np.random.random((17, 19, 19)).astype(DATA_DTYPE)
        policy = np.random.random((362,)).astype(DATA_DTYPE)
        value = np.float32(0.5)

        ts = TrainState(state, value, policy)

        packed = ts.pack()
        ts2 = TrainState.unpack(packed)

        self.assertTrue(np.allclose(ts2.state, ts.state))
        self.assertTrue(np.allclose(ts2.policy, ts.policy))
        self.assertTrue(np.allclose(ts2.value, ts.value))
        self.assertTrue(ts2.state.dtype == ts.state.dtype)
        self.assertTrue(ts2.policy.dtype == ts.policy.dtype)
        self.assertTrue(ts2.value.dtype == ts.value.dtype)
        self.assertTrue(ts2.state.shape == ts.state.shape)
        self.assertTrue(ts2.policy.shape == ts.policy.shape)
        self.assertTrue(ts2.value.shape == ts.value.shape)


class TestInference(unittest.TestCase):
    """Sanity test on Inference class"""

    def test_pack_unpack_inference(self):
        """Test pack and unpack inference methods"""
        inf = Inference(
            value=np.float32(0.5),
            policy=np.random.random((1, 362)).astype(DATA_DTYPE),
        )

        packed = inf.pack()
        inf2 = Inference.unpack(packed)

        self.assertTrue(np.allclose(inf2.value, inf.value))
        self.assertTrue(np.allclose(inf2.policy, inf.policy))
        self.assertTrue(inf2.value.dtype == inf.value.dtype)
        self.assertTrue(inf2.policy.dtype == inf.policy.dtype)
        self.assertTrue(inf2.value.shape == inf.value.shape)
        self.assertTrue(inf2.policy.shape == inf.policy.shape)

    def test_pack_unpack_value_array(self):
        """Test pack and unpack value array methods"""
        inf = Inference(
            value=np.random.random((1,)).astype(DATA_DTYPE),
            policy=np.random.random((1, 362)).astype(DATA_DTYPE),
        )

        packed = inf.pack()
        inf2 = Inference.unpack(packed)

        self.assertTrue(np.allclose(inf2.value, inf.value))
        self.assertTrue(np.allclose(inf2.policy, inf.policy))
        self.assertTrue(inf2.value.dtype == inf.value.dtype)
        self.assertTrue(inf2.policy.dtype == inf.policy.dtype)
        self.assertTrue(inf2.value.shape == inf.value.shape)
        self.assertTrue(inf2.policy.shape == inf.policy.shape)


class TestState(unittest.TestCase):
    """Sanity test on State class"""

    def test_pack_unpack(self):
        """Test pack and unpack state methods"""
        state = State(np.random.random((17, 19, 19)).astype(DATA_DTYPE))

        packed = state.pack()
        state2 = State.unpack(packed)

        self.assertTrue(np.allclose(state2.state, state.state))
        self.assertTrue(state2.state.dtype == state.state.dtype)
        self.assertTrue(state2.state.shape == state.state.shape)


if __name__ == "__main__":
    unittest.main()
