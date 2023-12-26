import numpy as np
from itertools import permutations
from scipy.stats import chi2
import unittest

class Test_MackSkil (unittest.TestCase):
    def test_two_implementations(self):
        # This verifies that the implementation of this class is the same as the implementation of normality.ipynb
        outp = {'k':6, 'n': 26, 'c':100}
        num_obs = outp['k'] * outp['n'] * outp['c']

        possible_ranks = np.tile(np.array(range(1,  outp['c']*outp['k']+1)), [outp['n'],1])
        mc_perm = np.zeros((outp['n'], outp['c']*outp['k']))
        for j in range(outp['n']):
            mc_perm[j, :] = np.random.permutation(possible_ranks[j, :])

        obs_data = mc_perm # Fake ranking data

        # Implementation 1
        S_vec = [np.sum(obs_data[:, (i-1)*outp['c']:(i*outp['c'])]) / outp['c'] for i in range(1, outp['k']+1)]
        MS_stat = 12 / (outp['k'] * (num_obs + outp['n'])) * np.sum(np.array(S_vec)**2) - 3 * (num_obs + outp['n'])
        print(f'implementation 1 : {MS_stat}')

        # Implementation 2 (from normality.ipynb)
        avg_rank = np.zeros((outp['n'], outp['k']))
        for i in range(outp['n']):
            for j in range(outp['k']):
                avg_rank[i, j] = np.mean(obs_data[i, j*outp['c']: j*outp['c']+outp['c']])

        S = np.mean(avg_rank, axis=0)*outp['n']
        N = num_obs
        MS = 12/(outp['k']*(N+outp['n']))*np.sum(np.power(S,2)) -3*(N+outp['n'])
        print(f'implementation 2 : {MS}')
        self.assertAlmostEqual(MS_stat, MS)


def cMackSkil(alpha, k, n, c, method=None, n_mc=10000):
    outp = {}
    outp["stat.name"] = "Mack-Skillings MS"
    outp["n.mc"] = n_mc
    outp["k"] = k
    outp["n"] = n
    outp["c"] = c

    if alpha > 1 or alpha < 0 or not isinstance(alpha, (int, float)):
        raise ValueError("Error: Check alpha value!")

    outp["alpha"] = alpha

    num_obs = outp["k"] * outp["n"] * outp["c"]

    if method is None:
        if np.math.factorial(outp["c"] * outp["k"] * outp["n"]) <= 100000:
            method = "Exact"
        else:  # Corresponds to: np.math.factorial(outp['c'] * outp['k']*outp['n']) > 10000
            method = "Monte Carlo"
    outp["method"] = method

    def MS_calc(obs_data):
        # print(obs_data)
        S_vec = [
            np.sum(obs_data[:, (i - 1) * outp["c"] : (i * outp["c"])]) / outp["c"]
            for i in range(1, outp["k"] + 1)
        ]
        MS_stat = 12 / (outp["k"] * (num_obs + outp["n"])) * np.sum(
            np.array(S_vec) ** 2
        ) - 3 * (num_obs + outp["n"])
        return MS_stat

    if outp["method"] == "Exact":
        possible_ranks = np.tile(
            np.array(range(1, outp["c"] * outp["k"] + 1)), [outp["n"], 1]
        )
        possible_perm = [
            np.reshape(arr, possible_ranks.shape)
            for arr in list(permutations(np.reshape(possible_ranks, -1)))
        ]
        # exact_dist = np.apply_along_axis(MS_calc, 1, possible_perm)
        exact_dist = list(map(MS_calc, list(possible_perm)))

        MS_vals = np.unique(exact_dist)
        MS_probs = np.array([np.sum(exact_dist == val) for val in MS_vals]) / (
            np.math.factorial(outp["c"] * outp["k"] * outp["n"])
        )
        MS_dist = np.column_stack((MS_vals, MS_probs))
        upper_tails = np.column_stack(
            (np.flip(MS_dist[:, 0]), np.cumsum(np.flip(MS_dist[:, 1])))
        )
        outp["cutoff_U"] = upper_tails[np.max(np.where(upper_tails[:, 1] <= alpha)), 0]
        outp["true_alpha_U"] = upper_tails[
            np.max(np.where(upper_tails[:, 1] <= alpha)), 1
        ]

    if outp["method"] == "Monte Carlo":
        possible_ranks = np.tile(
            np.array(range(1, outp["c"] * outp["k"] + 1)), [outp["n"], 1]
        )
        mc_perm = np.zeros((outp["n"], outp["c"] * outp["k"]))
        mc_stats = np.zeros(outp["n.mc"])
        for i in range(outp["n.mc"]):
            for j in range(outp["n"]):
                mc_perm[j, :] = np.random.permutation(possible_ranks[j, :])
            mc_stats[i] = round(MS_calc(mc_perm), 5)

        mc_vals = np.unique(mc_stats)
        mc_dist = np.array([np.sum(mc_stats == val) for val in mc_vals]) / outp["n.mc"]

        upper_tails = np.column_stack((np.flip(mc_vals), np.cumsum(np.flip(mc_dist))))
        outp["cutoff_U"] = upper_tails[np.max(np.where(upper_tails[:, 1] <= alpha)), 0]
        outp["true_alpha_U"] = upper_tails[
            np.max(np.where(upper_tails[:, 1] <= alpha)), 1
        ]

    if outp["method"] == "Asymptotic":
        outp["p_val"] = chi2.ppf(1 - alpha, outp["k"] - 1)

    return outp
