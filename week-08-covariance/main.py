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

def get_cov_matrix(data_matrix: np.typing.NDArray) -> np.typing.NDArray:
    X = data_matrix.copy()
    X = X - X.mean(axis=0)
    cov = (X.T @ X) / (X.shape[0] - 1)
    return cov

def get_eig_2x2(cov_matrix: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64] | None:
    return None

def main() -> None:
    # TESTING
    test_data = np.array([[1,2],[2,4],[3,6]])
    C = get_cov_matrix(test_data)
    assert np.allclose(C, [[1,2],[2,4]])
    assert np.allclose(C, np.cov(test_data, rowvar=False))
    print('cov checks passed')

if __name__ == '__main__':
    main()
