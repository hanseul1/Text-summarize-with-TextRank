import numpy as np

class TextRank(object):
    def get_rank(self, graph, d=0.85):
        A = graph
        matrix_size = A.shape[0]

        for i in range(matrix_size):
            A[i, i] = 0  # 대각선 부분 = 0
            link_sum = np.sum(A[:,i])  # A[:, i] = A[:][i]

            if link_sum != 0:
                A[:,i] /= link_sum
            A[:,i] *= -d
            A[i,i] = 1

        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)  # solve Ax = B
        
        return {idx: r[0] for idx, r in enumerate(ranks)}