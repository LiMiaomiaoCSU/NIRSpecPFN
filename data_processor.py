import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.cross_decomposition import PLSRegression
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from scipy.stats import ttest_1samp

class DataProcessor:
    def __init__(self, spectra, y):
        self.spectra = spectra
        self.y = y
        self.X_cal = None
        self.X_test = None
        self.y_cal = None
        self.y_test = None
        
    def load_and_split_data(self, method='random', test_size=0.25, random_state=42):
        if method == 'random':
            self.X_cal, self.X_test, self.y_cal, self.y_test = self.random(
                self.spectra, self.y, test_size=test_size, random_state=random_state
            )
        elif method == 'spxy':
            self.X_cal, self.X_test, self.y_cal, self.y_test = self.spxy(
                self.spectra, self.y, test_size=test_size
            )
        elif method == 'kennard-stone':
            self.X_cal, self.X_test, self.y_cal, self.y_test = self.ks(
                self.spectra, self.y, test_size=test_size
            )
        else:
            raise ValueError("未知的数据划分方法，请选择 'random', 'spxy', 或 'kennard-stone'。")
        print(f"数据集划分完成: 训练集大小 {self.X_cal.shape}, 测试集大小 {self.X_test.shape}")

    
    def random(self, data, label, test_size=0.25, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            data, label, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def spxy(self, data, label, test_size=0.25):
        x_backup = data
        y_backup = label
        M = data.shape[0]
        N = round((1 - test_size) * M)
        samples = np.arange(M)

        label = (label - np.mean(label)) / np.std(label)
        D = np.zeros((M, M))
        Dy = np.zeros((M, M))

        for i in range(M - 1):
            xa = data[i, :]
            ya = label[i]
            for j in range(i + 1, M):
                xb = data[j, :]
                yb = label[j]
                D[i, j] = np.linalg.norm(xa - xb)
                Dy[i, j] = np.linalg.norm(ya - yb)

        Dmax = np.max(D)
        Dymax = np.max(Dy)
        D = D / Dmax + Dy / Dymax

        maxD = D.max(axis=0)
        index_row = D.argmax(axis=0)
        index_column = np.argmax(maxD)

        m = np.zeros(N)
        m[0] = index_row[index_column]
        m[1] = index_column
        m = m.astype(int)

        dminmax = np.zeros(N)
        dminmax[1] = D[m[0], m[1]]

        for i in range(2, N):
            pool = np.delete(samples, m[:i])
            dmin = np.zeros(M - i)
            for j in range(M - i):
                indexa = pool[j]
                d = np.zeros(i)
                for k in range(i):
                    indexb = m[k]
                    if indexa < indexb:
                        d[k] = D[indexa, indexb]
                    else:
                        d[k] = D[indexb, indexa]
                dmin[j] = np.min(d)
            dminmax[i] = np.max(dmin)
            index = np.argmax(dmin)
            m[i] = pool[index]

        m_complement = np.delete(np.arange(data.shape[0]), m)

        X_train = data[m, :]
        y_train = y_backup[m]
        X_test = data[m_complement, :]
        y_test = y_backup[m_complement]

        return X_train, X_test, y_train, y_test

    def ks(self, data, label, test_size=0.25):
        M = data.shape[0]
        N = round((1 - test_size) * M)
        samples = np.arange(M)

        D = np.zeros((M, M))

        for i in range(M - 1):
            xa = data[i, :]
            for j in range(i + 1, M):
                xb = data[j, :]
                D[i, j] = np.linalg.norm(xa - xb)

        maxD = np.max(D, axis=0)
        index_row = np.argmax(D, axis=0)
        index_column = np.argmax(maxD)

        m = np.zeros(N)
        m[0] = index_row[index_column]
        m[1] = index_column
        m = m.astype(int)
        dminmax = np.zeros(N)
        dminmax[1] = D[m[0], m[1]]

        for i in range(2, N):
            pool = np.delete(samples, m[:i])
            dmin = np.zeros(M - i)
            for j in range(M - i):
                indexa = pool[j]
                d = np.zeros(i)
                for k in range(i):
                    indexb = m[k]
                    if indexa < indexb:
                        d[k] = D[indexa, indexb]
                    else:
                        d[k] = D[indexb, indexa]
                dmin[j] = np.min(d)
            dminmax[i] = np.max(dmin)
            index = np.argmax(dmin)
            m[i] = pool[index]

        m_complement = np.delete(np.arange(data.shape[0]), m)

        X_train = data[m, :]
        y_train = label[m]
        X_test = data[m_complement, :]
        y_test = label[m_complement]

        return X_train, X_test, y_train, y_test
    
    def select_train_samples(self, method='random', sample_size=10):
        """
        从当前的训练集中选择新的样本以扩展训练集。
        """
        if method == 'random':
            # 随机选择新的样本索引
            available_indices = [i for i in range(self.X_cal.shape[0]) if i not in self.selected_indices]
            new_indices = np.random.choice(available_indices, size=sample_size, replace=False)
        elif method == 'spxy':
            # 使用 SPXY 方法选择新的样本索引
            available_data = self.X_cal[[i for i in range(self.X_cal.shape[0]) if i not in self.selected_indices]]
            available_label = self.y_cal[[i for i in range(self.X_cal.shape[0]) if i not in self.selected_indices]]
            _, _, new_indices, _ = self.spxy(
                available_data, available_label, test_size=0.0  # 选择所有样本作为训练集
            )
            new_indices = new_indices[:sample_size]
        elif method == 'kennard-stone':
            # 使用 Kennard-Stone 方法选择新的样本索引
            available_data = self.X_cal[[i for i in range(self.X_cal.shape[0]) if i not in self.selected_indices]]
            available_label = self.y_cal[[i for i in range(self.X_cal.shape[0]) if i not in self.selected_indices]]
            _, _, new_indices, _ = self.ks(
                available_data, available_label, test_size=0.0  # 选择所有样本作为训练集
            )
            new_indices = new_indices[:sample_size]
        else:
            raise ValueError("未知的样本选择方法，请选择 'random', 'spxy', 或 'kennard-stone'。")
        
        # 将新选择的索引转换为原始数据的索引
        new_indices = [i for i in range(self.X_cal.shape[0]) if i not in self.selected_indices][:len(new_indices)]
        
        self.selected_indices.extend(new_indices)
        return self.X_cal[new_indices], self.y_cal[new_indices]
    
        
    def baseline_correction_airPLS(self, X, lambda_=100, itermax=15):
        """
        使用airPLS算法进行基线校正
        """
        corrected_X = np.zeros_like(X)
        for i in range(X.shape[0]):
            baseline = self.airPLS(X[i, :], lambda_=lambda_, itermax=itermax)
            corrected_X[i, :] = X[i, :] - baseline
        return corrected_X
    
    def airPLS(self, x, lambda_=100, porder=1, itermax=15):
        """
        airPLS算法实现
        x: 一维光谱数据
        lambda_: 平滑参数
        porder: 不对称权重
        itermax: 最大迭代次数
        """
        m = x.shape[0]
        w = np.ones(m)
        for i in range(1, itermax+1):
            z = self.WhittakerSmooth(x, w, lambda_, porder)
            d = x - z
            dssn = np.abs(d[d < 0].sum())
            if dssn < 0.001 * (abs(x)).sum() or i == itermax:
                if i == itermax:
                    print('airPLS到达最大迭代次数!')
                break
            w[d >= 0] = 0  # 正残差部分设为0
            w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
            w[0] = np.exp(i * (d[d < 0]).max() / dssn)
            w[-1] = w[0]
        return z
    
    def WhittakerSmooth(self, x, w, lambda_, differences=1):
        """
        Penalized least squares algorithm for background fitting
        """
        X = np.matrix(x)
        m = X.size
        E = sparse.eye(m, format='csc')
        D = E[1:] - E[:-1]  # 一阶差分
        W = sparse.diags(w, 0, shape=(m, m))
        A = sparse.csc_matrix(W + (lambda_ * D.T * D))
        B = sparse.csc_matrix(W * X.T)
        background = spsolve(A, B)
        return np.array(background)
    
    def perform_msc(self, X):
        """多元散射校正"""
        mean_spectrum = np.mean(X, axis=0)
        corrected_X = np.zeros_like(X)
        for i in range(X.shape[0]):
            # 线性回归拟合
            p = np.polyfit(mean_spectrum, X[i, :], 1)
            # 校正光谱
            corrected_X[i, :] = (X[i, :] - p[1]) / p[0]
        return corrected_X
    
    def perform_standard_normal_variate(self, X):
        """标准正态变量变换"""
        return (X - np.mean(X, axis=1)[:, np.newaxis]) / np.std(X, axis=1)[:, np.newaxis]
    
    def perform_savgol(self, X, window_length=5, polyorder=2, deriv=0):
        """Savitzky-Golay平滑滤波"""
        from scipy.signal import savgol_filter
        return savgol_filter(X, window_length, polyorder, deriv=deriv, axis=1)

    def perform_detrend(self, X):
        """去趋势预处理"""
        detrended_X = np.zeros_like(X)
        for i in range(X.shape[0]):
        # 对每个样本进行一阶多项式拟合
            x_axis = np.arange(X.shape[1])
            coeffs = np.polyfit(x_axis, X[i, :], 1)
        # 计算拟合的趋势线
            trend = np.polyval(coeffs, x_axis)
        # 去除趋势
            detrended_X[i, :] = X[i, :] - trend
        return detrended_X
    
    def spectral_first_order_derivative(self, X):
        """光谱一阶微分处理"""
        derivative_X = np.zeros_like(X)
        for i in range(X.shape[0]):
        # 对每个样本进行一阶微分处理
        # 使用中心差分法，两端点使用前向和后向差分
            derivative_X[i, 0] = X[i, 1] - X[i, 0]  # 前向差分
            for j in range(1, X.shape[1] - 1):
                derivative_X[i, j] = (X[i, j + 1] - X[i, j - 1]) / 2  # 中心差分
            derivative_X[i, -1] = X[i, -1] - X[i, -2]  # 后向差分
        return derivative_X
    
    def uniform_sampling(self, X_cal, X_test, count=100):
        """均匀采样"""
        n_features = X_cal.shape[1]
        step = max(1, n_features // count)
        indices = np.arange(0, n_features, step)
        return X_cal[:, indices], X_test[:, indices]
    
    def perform_univariant_selection(self, X, y, count=50):
        """单变量特征选择"""
        from sklearn.feature_selection import f_regression
        f_values, _ = f_regression(X, y)
        indices = np.argsort(f_values)[-count:]
        return indices
    
    def perform_recursive_elimination(self, X, y, count=50):
        """递归特征消除"""
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LinearRegression
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=count, step=0.1)
        selector.fit(X, y)
        return selector.support_
    
    def perform_pca(self, X, count=50):
        """主成分分析"""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=count)
        return pca.fit_transform(X)
    
    def perform_uve(self, X, y, count):
    # PLS回归
        pls = PLSRegression(n_components=min(10, X.shape[1], X.shape[0]))
        y_pred = cross_val_predict(pls, X, y, cv=min(5, len(y)))
        residuals = y - y_pred
    
    # 创建空数组存储结果
        stability = np.zeros(X.shape[1])
        relevance = np.zeros(X.shape[1])
    
    # 手动计算每一列，避免广播问题
        for i in range(X.shape[1]):
        # 获取当前特征列
            feature_col = X[:, i]
        
        # 计算每个样本的残差与特征的乘积
            products = np.zeros(len(residuals))
            for j in range(len(residuals)):
                products[j] = residuals[j] * feature_col[j]
        
        # 计算稳定性指标
            mean_val = np.mean(products)
            std_val = np.std(products)
        
            if std_val != 0:
                stability[i] = np.abs(mean_val / std_val)
            else:
                stability[i] = 0
            
        # 计算t统计量
            t_stat, _ = ttest_1samp(products, 0)
            relevance[i] = np.abs(t_stat) if not np.isnan(t_stat) else 0
    
    # 计算最终评分并返回索引
        scores = stability * relevance
        indices = np.argsort(scores)[-count:]
        return indices
