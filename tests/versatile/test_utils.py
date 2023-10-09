import unittest
import numpy as np
import cait.versatile as vai

class TestUtils(unittest.TestCase):
    def test_timestamp_coincidence_standard(self):
        a = np.array([10, 20, 30, 40])
        b = np.array([1, 11, 35, 42, 45])

        inside, coincidence_ind, outside = vai.utils.timestamp_coincidence(a, b, (-1,2))
        self.assertTrue(np.array_equal(inside, np.array([1])))
        self.assertTrue(np.array_equal(outside, np.array([0,2,3,4])))
        self.assertTrue(np.array_equal(coincidence_ind, np.array([0])))
        self.assertTrue(np.array_equal(b[inside], np.array([11])))
        self.assertTrue(np.array_equal(b[outside], np.array([1,35,42,45])))
        self.assertTrue(np.array_equal(a[coincidence_ind], np.array([10])))

        inside, coincidence_ind, outside = vai.utils.timestamp_coincidence(a, b, (-1,3))
        self.assertTrue(np.array_equal(inside, np.array([1,3])))
        self.assertTrue(np.array_equal(outside, np.array([0,2,4])))
        self.assertTrue(np.array_equal(coincidence_ind, np.array([0,3])))
        self.assertTrue(np.array_equal(b[inside], np.array([11,42])))
        self.assertTrue(np.array_equal(b[outside], np.array([1,35,45])))
        self.assertTrue(np.array_equal(a[coincidence_ind], np.array([10,40])))

    def test_timestamp_coincidence_all_negative(self):
        a = np.array([10, 20, 30, 40])
        b = np.array([7, 8, 11, 35, 42, 45])

        inside, coincidence_ind, outside = vai.utils.timestamp_coincidence(a, b, (-3,-1))
        self.assertTrue(np.array_equal(b[inside], np.array([7,8])))
        self.assertTrue(np.array_equal(b[outside], np.array([11,35,42,45])))
        self.assertTrue(np.array_equal(a[coincidence_ind], np.array([10,10])))

    def test_sample_noise(self): # TODO: implement
        ...

if __name__ == '__main__':
    unittest.main()