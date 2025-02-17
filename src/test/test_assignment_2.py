import unittest
from src.main.assignment_2 import neville, newton_forward, newton_interpolation, hermite_interpolation_fixed

class TestAssignment2(unittest.TestCase):

    def test_neville(self):
        x_vals = [3.6, 3.8, 3.9]
        y_vals = [1.675, 1.436, 1.318]
        result = neville(x_vals, y_vals, 3.7)
        self.assertAlmostEqual(result, 1.5549999999999995, places=6)

    def test_newton_forward(self):
        x_vals = [7.2, 7.4, 7.5, 7.6]
        y_vals = [23.5492, 25.3913, 26.8224, 27.4589]
        result = newton_forward(x_vals, y_vals)
        self.assertEqual(len(result), 3)

    def test_hermite(self):
        x_vals = [3.6, 3.8, 3.9]
        y_vals = [1.675, 1.436, 1.318]
        dy_vals = [-1.195, -1.188, -1.182]
        result = hermite_interpolation_fixed(x_vals, y_vals, dy_vals)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
