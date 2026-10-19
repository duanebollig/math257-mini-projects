import yfinance as yf
import numpy as np

PERIOD = '5y'
TICKER_1 = 'SPY'
TICKER_2 = 'QQQ'

def load_returns(first: str = TICKER_1, second: str = TICKER_2, period: str = PERIOD) -> np.typing.NDArray | None:
    f_ticker = yf.download(first, period=period)
    s_ticker = yf.download(second, period=period)
    # data_matrix = np.array()
    return None

def cov_matrix(data: np.typing.NDArray) -> np.typing.NDArray: # GET COVARIANCE FROM DATA MATRIX
    data = data - data.mean(axis=0)
    return (data.T @ data) / (data.shape[0] - 1)

def eigs_2x2(cov: np.typing.NDArray) -> tuple[np.typing.NDArray, np.typing.NDArray]: # COMPUTE EIGENVALUES AND EIGENVECTORS
    tr = (cov[0,0] + cov[1,1])
    det = (cov[0,0] * cov[1,1]) - (cov[0,1] * cov[1,0])
    # TRACE-DETERMINANT SHORCUT FOR EIGENVALUES
    disc = np.sqrt((tr ** 2) - (4 * det))
    eig_1 = (tr + disc) / 2
    eig_2 = (tr - disc) / 2
    # EIGENVECTORS
    vec_1 = np.array([-cov[0,1], cov[0,0] - eig_1])
    vec_1 = vec_1 / np.linalg.norm(vec_1)
    vec_2 = np.array([-cov[0,1], cov[0,0] - eig_2])
    vec_2 = vec_2 / np.linalg.norm(vec_2)

    return (np.array([eig_1, eig_2]), np.column_stack([vec_1, vec_2]))

def main() -> None:
    # HAND CHECKS
    hand_data = np.array([[1,2], [2,4], [3,6]])
    hand_cov = cov_matrix(hand_data)
    hand_vals, hand_v = eigs_2x2(hand_cov)
    assert np.allclose(hand_cov, [[1,2], [2,4]])
    assert np.allclose(hand_vals, [5,0])
    for i in range(2):
        assert np.allclose(hand_cov @ hand_v[:, i], hand_vals[i] * hand_v[:, i])
    assert np.allclose(hand_v.T @ hand_v, np.eye(2))
    print("all checks passed")



if __name__ == '__main__':
    main()


