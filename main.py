import numpy as np
from scipy.special import ellipk
from scipy.special import ellipj
import matplotlib.pyplot as plt


def pendulum_data(batch_size, noise, mode='single', Q=None):
    def sol(t, theta0):
        S = np.sin(0.5 * theta0)
        K_S = ellipk(S ** 2)
        omega_0 = np.sqrt(9.81)
        sn, cn, dn, ph = ellipj(K_S - omega_0 * t, S ** 2)
        theta = 2.0 * np.arcsin(S * sn)
        d_sn_du = cn * dn
        d_sn_dt = -omega_0 * d_sn_du
        d_theta_dt = 2.0 * S * d_sn_dt / np.sqrt(1.0 - (S * sn) ** 2)
        return np.stack([theta, d_theta_dt], axis=1)

    def compute_single_path(anal_ts, angle):
        # X = sol(anal_ts, .8)
        return sol(anal_ts, angle)

    # time samples and baseline angle
    anal_ts = np.arange(0, 1200 * 0.015, 0.015)
    # angle0 = args.angle
    angle0 = np.pi / 2
    # Rotate to high-dimensional space
    # if Q is None:
    #     Q = np.random.standard_normal((64, 2))
    #     Q, _ = np.linalg.qr(Q)
    if mode == 'single':
        X = compute_single_path(anal_ts, angle0)
        # generate additive noise
        N = noise * np.random.standard_normal((batch_size,) + X.shape)
    else:
        angles = angle0 + .3 * np.random.rand(batch_size)
        X = []
        for angle in angles:
            X.append(compute_single_path(anal_ts, angle))
        X = np.stack(X, axis=0)
        # generate additive noise
        N = noise * np.random.standard_normal((batch_size,) + X.shape[1:])
    # noisy 2D data
    Xn = X + N
    # # project to higher dim
    # Xu, Xnu = X @ Q.T, Xn @ Q.T
    # # scale
    # Xu = 2 * (Xu - np.min(Xu)) / np.ptp(Xu) - 1         # clean data
    # Xnu = 2 * (Xnu - np.min(Xnu)) / np.ptp(Xnu) - 1     # noisy data
    # # translate
    # Xu = Xu + (2 * np.random.rand(Xu.shape[-1]) - 1)
    # Xnu = Xnu + (2* np.random.rand(Xnu.shape[-1]) - 1)
    return Xn


def myDMD(Z):
    # organizing data
    X = Z[:len(Z) - 1].T
    Y = Z[1:len(Z)].T
    # computing svd
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    smat = np.diag(s)  # np.allclose(X,u@smat@vh)==True
    # define Atilda=(U^h)(Y)(V)(Sigma^-1)
    Atilde = u.T @ Y @ vh.T @ np.linalg.inv(smat)
    # finding Atilda eigenvectors
    vals, ws = np.linalg.eig(Atilde)  # np.allclose(Atilda@ws,ws@np.diag(vals))==True
    # calculate the modes
    modes = Y @ vh.T @ (np.linalg.inv(smat)) @ ws
    return modes, smat


def predict(modes, smat, b0=1, t=1):
    return modes @ (smat ** t) * b0

def accuracy():
    x=5
if __name__ == '__main__':
    # generating data
    Xn = pendulum_data(batch_size=1, noise=0)
    # plotting some of the data
    path_len = 400
    train_size = 700
    # x_p = Xn[0][:path_len, 0]
    # y_p = Xn[0][:path_len, 1]
    # plt.scatter(x_p, y_p, s=1)
    # plt.show()
    # calling DMD func
    train = Xn[0][:train_size, :]
    test = Xn[0][train_size:, :]
    modes, smat = myDMD(train)
    # check accuracy
    t=50 # how far to predict
    b0=1
    Yt = predict(modes, smat,t=t,b0=b0)
    # plot prediction vs reality
    plt.scatter(Yt[t:,0], Yt[t:,1],c='green' ,s=20)
    plt.scatter(test[:t,0], test[:t,1],c='blue' ,s=20)


    plt.show()
    # print(f"t= {t}, b0= {b0}, accuracy= {}")
    x=3