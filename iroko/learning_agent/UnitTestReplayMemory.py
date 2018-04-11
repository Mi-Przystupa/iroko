import unittest
import torch
from ReplayMemory import ReplayMemory


class UnitTestReplayMemory(unittest.TestCase):

    def testDefaultInit(self):
        RepMem = ReplayMemory()
        self.assertEqual(RepMem.size, 40)

    def testPush(self):
        RepMem = ReplayMemory()
        actions = ['a', 'b', 'c']
        for i in range(0, RepMem.size):
            RepMem.push(i, actions[i % 3], 1, i + 1)
            memory = RepMem.memory[i]
            self.assertEqual(memory['a'], actions[i % 3])
            self.assertEqual(memory['s'], i)
            self.assertEqual(memory['sprime'], i + 1)
            self.assertEqual(memory['r'], 1)

        for i in range(RepMem.size, RepMem.size * 2):
            RepMem.push(i, actions[i % 3], 1, i + 1)
            memory = RepMem.memory[i % RepMem.size]
            self.assertEqual(memory['a'], actions[i % 3])
            self.assertEqual(memory['s'], i)
            self.assertEqual(memory['sprime'], i + 1)
            self.assertEqual(memory['r'], 1)

    def testPushInxes(self):
        RepMem = ReplayMemory()

        for i in range(0, 200000):
            RepMem.push(1, 1, 1, 1)
            self.assertEqual(RepMem.indx, ((i + 1) % RepMem.size))
        self.assertTrue(RepMem.isFull())


if __name__ == '__main__':
    unittest.main()
