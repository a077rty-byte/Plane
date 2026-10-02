# =========================================================
# Chaotic BPNN (BSLC) 
#   Input : A (inputs), B (hidden), C (outputs), P (samples), eta, eps, mu,
#           x0, y0, [a, b], [theta_min, theta_max], [alpha_min, alpha_max]
#   Output: W1[A x B], W2[B x C]
# =========================================================
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

SEED = 42
np.random.seed(SEED)          # used only for shuffling the samples, never for weights

# ---------------- Dataset config ----------------
CSV_PATH   = r"/content/ .csv"
TARGET_COL = " "

# ---------------- Algorithm inputs ----------------
B          = 128        # number of hidden nodes
C          = 1          # number of output nodes 
ETA        = 0.1        # learning rate  (eta)
EPSILON    = 0.005      # acceptable error threshold (eps)
MAX_EPOCHS = 200        # safety cap on "While E > eps"
BATCH_SIZE = 1          # 1 = "For each sample p"
                        # >1  = mini-batch

MU_BSLC    = 0.10       # BSLC coupling parameter mu
X0, Y0     = 0.2468, 0.3691   # initial chaotic seeds
BURN_IN    = 200        # discard transient iterations

A_WEIGHT, B_WEIGHT = -1.0, 1.0     # weight range [a, b]
THETA_MIN, THETA_MAX = 0.0, 0.1    # chaotic bias range
ALPHA_MIN, ALPHA_MAX = 0.98, 0.99999  # chaotic momentum range


# =========================================================
# BSLC chaotic stream: one continuous generator
#   x_{n+1} = 4x_n(1-x_n)(1-mu) + mu sin^2(pi y_n / 2)
#   y_{n+1} = 4y_n(1-y_n)(1-mu) + mu sin^2(pi x_n / 2)
#   Z_n = (x_n * y_n) mod 1
# =========================================================
class BSLCStream:
    def __init__(self, mu=MU_BSLC, x0=X0, y0=Y0, burn=BURN_IN):
        self.mu, self.x, self.y = mu, float(x0), float(y0)
        for _ in range(burn):
            self._step()

    def _step(self):
        mu, x, y = self.mu, self.x, self.y
        x_new = 4.0*x*(1.0-x)*(1.0-mu) + mu*np.sin(np.pi*y/2.0)**2
        y_new = 4.0*y*(1.0-y)*(1.0-mu) + mu*np.sin(np.pi*x/2.0)**2
        # keep the orbit inside (0,1) so Z_n in (0,1)
        self.x = float(np.clip(x_new, 1e-7, 1-1e-7))
        self.y = float(np.clip(y_new, 1e-7, 1-1e-7))

    def next_z(self):
        """One BSLC iteration -> projection Z_n = (x_n * y_n) mod 1."""
        self._step()
        return (self.x * self.y) % 1.0

    def z_sequence(self, n):
        return np.array([self.next_z() for _ in range(n)], dtype=np.float64)


# =========================================================
# Other Z_n sources, used only for WEIGHT initialization (InitMethod).
# theta_n and alpha_n always come from BSLC, as the algorithm states.
# =========================================================
class _Map1D:
    """1D map with the same interface as BSLCStream."""
    def __init__(self, f, x0, burn, to_z=lambda x: x):
        self.f, self.x, self.to_z = f, float(x0), to_z
        for _ in range(burn):
            self.x = self.f(self.x)

    def next_z(self):
        self.x = self.f(self.x)
        return float(np.clip(self.to_z(self.x), 1e-7, 1-1e-7))

    def z_sequence(self, n):
        return np.array([self.next_z() for _ in range(n)], dtype=np.float64)

def logistic_stream(r=3.9, x0=0.123456, burn=BURN_IN):
    return _Map1D(lambda x: float(np.clip(r*x*(1.0-x), 1e-7, 1-1e-7)), x0, burn)

def tanh_stream(x0=0.111, burn=50):
    return _Map1D(lambda x: float(np.tanh(2.0*x + 0.1)), x0, burn, to_z=lambda x: 0.5*(x+1.0))

def sigmoid_stream(x0=0.111, burn=50):
    return _Map1D(lambda x: float(1.0/(1.0+np.exp(-(4.0*x-2.0)))), x0, burn)

def make_init_stream(method):
    if method == "tanh":     return tanh_stream()
    if method == "sigmoid":  return sigmoid_stream()
    if method == "BSLC":     return BSLCStream()
    if method == "Logistic": return logistic_stream()
    raise ValueError(f"Unknown init method: {method}")


# =========================================================
# 1. Chaotic initialization
# =========================================================
def chaotic_init(stream, A, B, C, a=A_WEIGHT, b=B_WEIGHT):
    # W1 and W2 take consecutive (not repeated) values of the same sequence
    Z = stream.z_sequence(A*B + B*C)
    U = 2.0*Z - 1.0                         # U_n = 2Z_n - 1,  U_n in [-1, 1]
    W = a + (b - a)*(U + 1.0)/2.0           # maps U_n in [-1,1] onto [a, b]  (see note 1)
    W1 = W[:A*B].reshape(A, B)
    W2 = W[A*B:].reshape(B, C)
    return W1, W2

def theta_from_z(z):                        # theta_n = theta_min + (theta_max-theta_min) Z_n
    return THETA_MIN + (THETA_MAX - THETA_MIN)*z

def alpha_from_z(z):                        # alpha_n = alpha_min + (alpha_max-alpha_min) Z_n
    return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN)*z


# =========================================================
# Activation phi (sigmoid) and its derivative phi'
# =========================================================
def phi(v):
    return 1.0/(1.0 + np.exp(-np.clip(v, -500, 500)))

def dphi_from_out(out):                     # phi'(net) = phi(net)(1 - phi(net))
    return out*(1.0 - out)


def forward(X, W1, W2, theta):
    net_j = X @ W1 + theta                  # net_j = sum_i W1_ij x_i + theta_n
    h     = phi(net_j)                      # h_j = phi(net_j)
    net_k = h @ W2                          # net_k = sum_j W2_jk h_j
    a     = phi(net_k)                      # a_k = phi(net_k)
    return h, a


# =========================================================
#   use_alpha=True  -> proposed method: W <- alpha_n W + eta delta (.)
#   use_alpha=False -> baseline GBDR:   W <- W + eta delta (.)    (alpha = 1)
# =========================================================
def train(X, T, B=B, C=C, eta=ETA, eps=EPSILON, max_epochs=MAX_EPOCHS,
          batch_size=BATCH_SIZE, use_alpha=True, init_method="BSLC", verbose=True):
    P, A = X.shape
    T = T.reshape(-1, C).astype(np.float64)
    stream = BSLCStream()                       # source of theta_n and alpha_n

# BSLC: one continuous stream
    init_stream = stream if init_method == "BSLC" else make_init_stream(init_method)
    W1, W2 = chaotic_init(init_stream, A, B, C)
    iters = 0
    z      = stream.next_z()                # Z_n for the first update
    theta, alpha = theta_from_z(z), alpha_from_z(z)

    rng = np.random.RandomState(SEED)
    E, epoch, history = np.inf, 0, []
    t0 = time.time()

    while E > eps and epoch < max_epochs:   # While E > eps
        order = rng.permutation(P)
        sq_err = 0.0
        for s in range(0, P, batch_size):   # For each sample p (or mini-batch)
            idx = order[s:s+batch_size]
            x, t = X[idx], T[idx]

            h, a = forward(x, W1, W2, theta)

            err = t - a
            sq_err += 0.5*np.sum(err**2)    # accumulate 1/2 sum (t - a)^2

            delta_k = err*dphi_from_out(a)                    # delta_k = (t_k - a_k) phi'(net_k)
            delta_j = dphi_from_out(h)*(delta_k @ W2.T)       # delta_j = phi'(net_j) sum_k delta_k W2_jk

            m = len(idx)                    # averaged over the batch (m=1 -> identical to online rule)
            a_n = alpha if use_alpha else 1.0
            W2 = a_n*W2 + eta*(h.T @ delta_k)/m               # W2 <- alpha_n W2 + eta delta_k h_j
            W1 = a_n*W1 + eta*(x.T @ delta_j)/m               # W1 <- alpha_n W1 + eta delta_j x_i
            iters += 1

            z = stream.next_z()             # Update Z_n -> theta_n, alpha_n from next BSLC iteration
            theta, alpha = theta_from_z(z), alpha_from_z(z)

        E = sq_err / P                      # E = 1/P sum_p 1/2 sum_k (t - a)^2
        epoch += 1
        history.append(E)
        if verbose:
            print(f"  epoch {epoch:3d} | E = {E:.6f} | theta = {theta:.4f} | alpha = {alpha:.5f}")

    info = dict(epochs=epoch, iters=iters, final_E=E, time=time.time()-t0,
                converged=bool(E <= eps), history=history, theta_last=theta)
    return W1, W2, info                     # Return W1, W2


# =========================================================
# Evaluation (outside the algorithm)
# =========================================================
def predict_proba(X, W1, W2, theta):
    return forward(X, W1, W2, theta)[1].ravel()

def best_threshold_f1(y_true, y_prob):
    ts = np.arange(0.0, 1.0001, 0.01)
    f  = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f))])

def evaluate(y_true, y_prob, thr):
    yh = (y_prob >= thr).astype(int)
    return (accuracy_score(y_true, yh), precision_score(y_true, yh, zero_division=0),
            recall_score(y_true, yh, zero_division=0), f1_score(y_true, yh, zero_division=0),
            confusion_matrix(y_true, yh))


# =========================================================
# Data loading 
# =========================================================
def load_data(path=CSV_PATH, target=TARGET_COL):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")
    df = df.drop(columns=[c for c in ["Flow ID","Timestamp","Src IP","Dst IP",
                                      "Source IP","Destination IP"] if c in df.columns])
    n0 = len(df)
    df = df[df[target].astype(str).str.strip().isin(['0', '1'])]
    print(f"Removed {n0 - len(df)} rows with invalid '{target}' values.")

    X_df = df.drop(columns=[target]).copy()
    for c in X_df.columns:
        if X_df[c].dtype == object:
            X_df[c] = pd.factorize(X_df[c].astype(str))[0].astype(np.float64)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)
    X = X_df.values.astype(np.float64)
    y = df[target].astype(str).str.strip().astype(np.int32).values
    return X, y


# Print order and labels, as in the results table
ORDERED_INITS = ["tanh", "sigmoid", "BSLC", "Logistic"]
MODES = [("GBDR", False), ("Chaotic momentum (\u03b1)", True)]
TASK = "binary"
ACT_LABEL = "BSLC \u03c6(s)"   # label only; phi in this code is sigmoid. Change it if that's wrong


def run_experiment(X, y, verbose=False):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                              random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.20,
                                              random_state=SEED, stratify=y_tr)
    scaler = MinMaxScaler().fit(X_tr)                  # fit on training data only
    X_tr, X_va, X_te = (np.clip(scaler.transform(Z), 0, 1) for Z in (X_tr, X_va, X_te))
    print(f"A = {X_tr.shape[1]} | Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

    rows, cms = [], []
    for init_m in ORDERED_INITS:
        for mode, use_alpha in MODES:
            if verbose:
                print(f"\n--- Init={init_m} | Mode={mode} ---")
            W1, W2, info = train(X_tr, y_tr, use_alpha=use_alpha,
                                 init_method=init_m, verbose=verbose)
            th = info["theta_last"]
            thr = best_threshold_f1(y_va, predict_proba(X_va, W1, W2, th))
            acc, prec, rec, f1v, cm = evaluate(y_te, predict_proba(X_te, W1, W2, th), thr)
            rows.append((init_m, mode, acc, prec, rec, f1v, info["time"], info["iters"], thr))
            cms.append((init_m, mode, cm))

    print(f"\n==== Intrusion (Activation = {ACT_LABEL}) | TASK={TASK} ====")
    print("InitMethod | Mode                   | Acc    | Prec   | Recall | F1     | Time(s) | Iter   | BestThr")
    print("-"*112)
    for init_m, mode, acc, prec, rec, f1v, tsec, iters, thr in rows:
        print(f"{init_m:10s} | {mode:22s} | {acc:6.4f} | {prec:6.4f} | {rec:6.4f} | "
              f"{f1v:6.4f} | {tsec:7.2f} | {iters:6d} | {thr:6.3f}")

    print("\n==== Confusion Matrices (Test) ====")
    for init_m, mode, cm in cms:
        print(f"\n--- Init={init_m} | Mode={mode} ---")
        print(cm)
    return rows


if __name__ == "__main__":
    X, y = load_data()
    run_experiment(X, y)


