import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.colors import to_rgb

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
except Exception:
    class BaseEstimator:
        def get_params(self, deep=True):
            return {
                k: v
                for k, v in self.__dict__.items()
                if not k.endswith("_") and not k.startswith("_")
            }

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class ClassifierMixin:
        pass

    class RegressorMixin:
        pass

from ._core import LicketySPLIT as _LicketySPLITCore
from ._core import CacheMode
from ._threshold_guessing import ThresholdGuessBinarizer

__all__ = [
    "LicketySPLIT",
    "LicketySPLITClassifier",
    "LicketySPLITRegressor",
    "CacheMode",
    "ThresholdGuessBinarizer",
]


def _as_cache_mode(cache_mode):
    if isinstance(cache_mode, str):
        s = cache_mode.strip().lower().replace("-", "_").replace(" ", "_")

        if s in {"fingerprint", "hash", "hash_fingerprint"}:
            return CacheMode.HASH_FINGERPRINT

        if s in {"bitvector", "bit_vector", "bits", "exact"}:
            return CacheMode.BITVECTOR

        raise ValueError(
            "cache_mode must be one of "
            "{'fingerprint', 'hash', 'hash_fingerprint', "
            "'bitvector', 'bit_vector', 'bits', 'exact'}"
        )

    return cache_mode


def _as_u8_X(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.uint8)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")
    return np.ascontiguousarray(X)


def _as_i32_y(y, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int32)
    if y.ndim != 1:
        raise ValueError("y must be 1D (n_samples,)")
    if y.shape[0] != n:
        raise ValueError("y length must match X rows")
    if np.any(y < 0):
        raise ValueError("y labels must be nonnegative contiguous integers 0,1,2,...,C-1")

    if y.size:
        max_y = int(y.max())
        present = np.zeros(max_y + 1, dtype=bool)
        present[y] = True
        if not bool(np.all(present)):
            raise ValueError("y labels must be contiguous integers 0,1,2,...,C-1")

    return np.ascontiguousarray(y)


def _as_i32_vec(v, n: int, name: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.int32)
    if v.ndim != 1:
        raise ValueError(f"{name} must be 1D (n_samples,)")
    if v.shape[0] != n:
        raise ValueError(f"{name} length must match X rows")
    return np.ascontiguousarray(v)


def _as_f64_vec(v, n: int, name: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"{name} must be 1D (n_samples,)")
    if v.shape[0] != n:
        raise ValueError(f"{name} length must match X rows")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(v)


def _as_f64_weights(sample_weight, n: int) -> np.ndarray:
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if sample_weight.ndim != 1:
        raise ValueError("sample_weight must be 1D (n_samples,)")
    if sample_weight.shape[0] != n:
        raise ValueError("sample_weight length must match X rows")
    if not np.all(np.isfinite(sample_weight)):
        raise ValueError("sample_weight must be finite")
    if np.any(sample_weight < 0):
        raise ValueError("sample_weight must be nonnegative")
    return np.ascontiguousarray(sample_weight)


def _huge_eta_no_defer(n_rows: int, n_features: int, sample_weight) -> float:
    if sample_weight is None:
        sum_weights = float(n_rows)
    else:
        sum_weights = float(np.sum(sample_weight))

    return 1000.0 * (float(n_rows) * sum_weights * float(n_features)) + 1000.0


class _BaseLicketySPLIT(BaseEstimator):
    def _new_core(self):
        model = _LicketySPLITCore()
        model.set_cache_mode(_as_cache_mode(self.cache_mode))
        model.set_cost_caching_enabled(bool(self.cost_caching_enabled))
        return model

    def set_cache_mode(self, mode="fingerprint"):
        self.cache_mode = mode
        if hasattr(self, "_model"):
            self._model.set_cache_mode(_as_cache_mode(mode))
        return self

    def set_cost_caching_enabled(self, enabled: bool):
        self.cost_caching_enabled = bool(enabled)
        if hasattr(self, "_model"):
            self._model.set_cost_caching_enabled(bool(enabled))
        return self

    def clear_cost_caches(self):
        if hasattr(self, "_model"):
            self._model.clear_cost_caches()
        return self

    @property
    def last_objective(self) -> float:
        return float(self._model.last_objective())

    def split_counts(self, X) -> np.ndarray:
        X = _as_u8_X(X)
        return np.asarray(self._model.split_counts_single_tree(X), dtype=np.int32)


class LicketySPLITClassifier(_BaseLicketySPLIT, ClassifierMixin):
    def __init__(
        self,
        *,
        lambda_leaf: float = 0.01,
        depth_budget: int = 3,
        lookahead_k: int = 1,
        eta_defer=None,
        cache_mode="fingerprint",
        cost_caching_enabled: bool = True,
    ):
        self.lambda_leaf = lambda_leaf
        self.depth_budget = depth_budget
        self.lookahead_k = lookahead_k
        self.eta_defer = eta_defer
        self.cache_mode = cache_mode
        self.cost_caching_enabled = cost_caching_enabled

    def fit(
        self,
        X,
        y,
        *,
        bb_pred=None,
        sample_weight=None,
    ):
        X = _as_u8_X(X)
        n = int(X.shape[0])
        n_features = int(X.shape[1])

        y = _as_i32_y(y, n)

        weights_arg = None
        if sample_weight is not None:
            weights_arg = _as_f64_weights(sample_weight, n)

        if bb_pred is None:
            bb_arg = y.astype(np.int32, copy=True)
            eta_arg = _huge_eta_no_defer(n, n_features, weights_arg)
            self.uses_defer_ = False
        else:
            bb_arg = _as_i32_vec(bb_pred, n, "bb_pred")
            if np.any(bb_arg < 0):
                raise ValueError("bb_pred values must be nonnegative")
            if y.size:
                max_y = int(y.max())
                if np.any(bb_arg > max_y):
                    raise ValueError("bb_pred values must be in the same class range as y")
            eta_arg = 0.0 if self.eta_defer is None else float(self.eta_defer)
            self.uses_defer_ = True

        self._model = self._new_core()

        obj = self._model.fit(
            X,
            y,
            bb_arg,
            float(self.lambda_leaf),
            float(eta_arg),
            int(self.depth_budget),
            int(self.lookahead_k),
            weights_arg,
        )

        self.objective_ = float(obj)
        self.n_features_in_ = n_features
        self.classes_ = np.arange(int(y.max()) + 1, dtype=np.int32) if y.size else np.array([], dtype=np.int32)
        self.n_classes_ = int(len(self.classes_))
        self.eta_defer_used_ = float(eta_arg)

        self._compute_leaf_proba_(X, y, sample_weight)

        return self

    def fit_return(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return float(self.objective_)

    def _leaf_ids_from_paths(self, X) -> np.ndarray:
        # return the index of the leaf path reached by each row of X.
        X = _as_u8_X(X)
        n = int(X.shape[0])

        paths, _ = self.get_tree_paths()
        leaf_ids = np.full(n, -1, dtype=np.int32)

        for leaf_id, path in enumerate(paths):
            mask = np.ones(n, dtype=bool)

            for signed_f in path:
                f = abs(int(signed_f)) - 1

                if signed_f > 0:
                    mask &= (X[:, f] == 0)
                else:
                    mask &= (X[:, f] != 0)

            leaf_ids[mask] = int(leaf_id)

        if np.any(leaf_ids < 0):
            raise RuntimeError("Some samples did not match any leaf path.")

        return leaf_ids

    def _compute_leaf_proba_(self, X, y, sample_weight=None):
        # compute empirical class proportions in each fitted leaf.
        leaf_ids = self._leaf_ids_from_paths(X)
        n_leaves = int(leaf_ids.max()) + 1 if leaf_ids.size else 0
        n_classes = int(self.n_classes_)

        counts = np.zeros((n_leaves, n_classes), dtype=np.float64)

        if sample_weight is None:
            for leaf_id, yi in zip(leaf_ids, y):
                counts[int(leaf_id), int(yi)] += 1.0
        else:
            w = _as_f64_weights(sample_weight, int(X.shape[0]))
            for leaf_id, yi, wi in zip(leaf_ids, y, w):
                counts[int(leaf_id), int(yi)] += float(wi)

        totals = counts.sum(axis=1, keepdims=True)

        # Should not happen for fitted leaves, but safe fallback.
        zero = totals[:, 0] <= 0.0
        totals[zero, 0] = 1.0

        proba = counts / totals

        # If a zero-support leaf somehow exists, use the global class distribution.
        if np.any(zero):
            global_counts = counts.sum(axis=0)
            global_total = float(global_counts.sum())
            if global_total > 0.0:
                proba[zero, :] = global_counts / global_total
            else:
                proba[zero, :] = 1.0 / float(n_classes)

        self.leaf_proba_ = proba
        return self

    def predict(self, X, *, bb_pred=None) -> np.ndarray:
        X = _as_u8_X(X)
        n = int(X.shape[0])

        if not hasattr(self, "_model"):
            raise RuntimeError("Model has not been fit.")

        if getattr(self, "uses_defer_", False):
            if bb_pred is None:
                raise ValueError("This model was fit with deferral; pass bb_pred to predict.")
            bb_arg = _as_i32_vec(bb_pred, n, "bb_pred")
        else:
            bb_arg = np.zeros(n, dtype=np.int32)

        return np.asarray(self._model.predict(X, bb_arg), dtype=np.int32)

    def predict_placeholder(self, X, placeholder=99) -> np.ndarray:
        X = _as_u8_X(X)
        return np.asarray(self._model.predict_placeholder(X, int(placeholder)), dtype=np.int32)

    def predict_proba(self, X) -> np.ndarray:
        # return empirical class probabilities for the leaf reached by each row.
        # (n_samples, n_classes) ordered according to self.classes_.
        X = _as_u8_X(X)

        if not hasattr(self, "_model"):
            raise RuntimeError("Model has not been fit.")

        if not hasattr(self, "leaf_proba_"):
            raise RuntimeError("Leaf probabilities have not been computed.")

        leaf_ids = self._leaf_ids_from_paths(X)
        return np.asarray(self.leaf_proba_[leaf_ids], dtype=np.float64)

    def score(self, X, y, sample_weight=None):
        X = _as_u8_X(X)
        y = _as_i32_y(y, int(X.shape[0]))
        pred = self.predict(X)

        correct = (pred == y).astype(np.float64)

        if sample_weight is None:
            return float(np.mean(correct))

        w = _as_f64_weights(sample_weight, int(X.shape[0]))
        denom = float(np.sum(w))
        if denom <= 0.0:
            return 0.0
        return float(np.sum(w * correct) / denom)

    def leaf_counts(self) -> dict:
        return dict(self._model.leaf_counts_single_tree())

    def leaf_paths(self) -> list[list[int]]:
        return list(self._model.leaf_paths_single_tree())

    def leaf_actions(self) -> list[int]:
        return list(self._model.leaf_actions_single_tree())

    def get_tree_paths(self):
        paths, actions = self._model.leaf_paths_and_actions_single_tree()
        return paths, actions

    def plot_tree(
        self,
        feature_names=None,
        figsize=(14, 9),
        ax=None,
        title=None,
        show=True,
        class_color_low="#dbeafe",
        class_color_high="#fecaca",
        split_node_color="#e5e7eb",
        edge_color="black",
        node_scale=1.35,
        x_scale=5.8,
        y_scale=4.3,
    ):
        paths, actions = self.get_tree_paths()

        class Node:
            __slots__ = ("feature", "left", "right", "action")

            def __init__(self):
                self.feature = None
                self.left = None
                self.right = None
                self.action = None

        def build_tree_from_paths(paths, actions):
            root = Node()
            for path, act in zip(paths, actions):
                cur = root
                for signed_f in path:
                    f = abs(signed_f) - 1
                    go_left = signed_f > 0
                    if cur.feature is None:
                        cur.feature = f
                        cur.left = Node()
                        cur.right = Node()
                    cur = cur.left if go_left else cur.right
                cur.action = int(act)
            return root

        def collect_leaves_in_order(node, leaves):
            if node is None:
                return
            if node.action is not None or (node.left is None and node.right is None):
                leaves.append(node)
                return
            collect_leaves_in_order(node.left, leaves)
            collect_leaves_in_order(node.right, leaves)

        def assign_positions_tree(root):
            positions = {}
            leaves = []
            collect_leaves_in_order(root, leaves)
            if not leaves:
                leaves = [root]
            leaf_x = {leaf: i for i, leaf in enumerate(leaves)}

            def dfs(node, depth):
                if node is None:
                    return
                if node.action is not None or (node.left is None and node.right is None):
                    x = leaf_x[node]
                    positions[node] = (x, -depth)
                    return
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)
                x_left, _ = positions[node.left]
                x_right, _ = positions[node.right]
                positions[node] = (0.5 * (x_left + x_right), -depth)

            dfs(root, 0)
            return positions, len(leaves)

        def _shrink_segment(x1, y1, x2, y2, r1, r2):
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx * dx + dy * dy) ** 0.5
            if dist == 0:
                return x1, y1, x2, y2
            ux = dx / dist
            uy = dy / dist
            return (x1 + ux * r1, y1 + uy * r1, x2 - ux * r2, y2 - uy * r2)

        def _lerp_color(c1, c2, t):
            a = np.array(to_rgb(c1), dtype=float)
            b = np.array(to_rgb(c2), dtype=float)
            out = (1.0 - t) * a + t * b
            return tuple(out)

        def _class_color(action, n_classes):
            if n_classes <= 1:
                return class_color_low
            t = float(action) / float(n_classes - 1)
            t = max(0.0, min(1.0, t))
            return _lerp_color(class_color_low, class_color_high, t)

        if feature_names is None:
            enc = [abs(v) for p in paths for v in p]
            max_f = (max(enc) - 1) if enc else -1
            feature_names = [f"f{j}" for j in range(max_f + 1)]

        root = build_tree_from_paths(paths, actions)
        pos_local, _ = assign_positions_tree(root)

        pos = {}
        xs = []
        ys = []
        for node, (x, y) in pos_local.items():
            xx = x * x_scale
            yy = y * y_scale
            pos[node] = (xx, yy)
            xs.append(xx)
            ys.append(yy)

        minx, maxx = min(xs), max(xs)
        x_center = 0.5 * (minx + maxx)
        for node, (xx, yy) in list(pos.items()):
            pos[node] = (xx - x_center, yy)

        if ax is None:
            all_nodes_xy = list(pos.values())
            xs = [p[0] for p in all_nodes_xy] if all_nodes_xy else [0.0]
            ys = [p[1] for p in all_nodes_xy] if all_nodes_xy else [0.0]
            total_w = (max(xs) - min(xs)) if xs else 10.0
            total_h = (max(ys) - min(ys)) if ys else 6.0
            fig_w = max(figsize[0], total_w / 2.0)
            fig_h = max(figsize[1], total_h / 1.45)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        else:
            fig = ax.figure

        ax.set_axis_off()

        internal_r = 0.62 * node_scale
        leaf_r = 0.76 * node_scale
        edge_lw = 2.8 * node_scale

        SPLIT_FS = int(17 * node_scale)
        LEAF_FS = int(18 * node_scale)
        TITLE_FS = int(20 * node_scale)
        SPLIT_OFFSET_PTS = 16

        n_classes = 1
        if actions:
            n_classes = max(int(a) for a in actions if int(a) >= 0) + 1

        def _feat_name(f: int) -> str:
            if feature_names is not None and f < len(feature_names):
                return str(feature_names[f])
            return f"f{f}"

        def draw(node):
            x, y = pos[node]

            if node.feature is not None:
                if node.left is not None:
                    x2, y2 = pos[node.left]
                    r_child = leaf_r if node.left.action is not None else internal_r
                    sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, internal_r, r_child)
                    ax.add_line(Line2D([sx, ex], [sy, ey], color=edge_color, linewidth=edge_lw, zorder=1))
                    draw(node.left)

                if node.right is not None:
                    x2, y2 = pos[node.right]
                    r_child = leaf_r if node.right.action is not None else internal_r
                    sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, internal_r, r_child)
                    ax.add_line(Line2D([sx, ex], [sy, ey], color=edge_color, linewidth=edge_lw, zorder=1))
                    draw(node.right)

            if node.action is None:
                radius = internal_r
                circ = Circle(
                    (x, y),
                    radius,
                    facecolor=split_node_color,
                    edgecolor=edge_color,
                    linewidth=1.9,
                    zorder=10,
                )
                ax.add_patch(circ)

                if node.feature is not None:
                    name = _feat_name(int(node.feature))
                    ax.annotate(
                        name,
                        xy=(x, y),
                        xycoords="data",
                        xytext=(0, SPLIT_OFFSET_PTS),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=SPLIT_FS,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.98),
                        zorder=10000,
                        clip_on=False,
                    )
            else:
                if int(node.action) < 0:
                    face = "#fef3c7"
                    label = "defer"
                else:
                    face = _class_color(int(node.action), n_classes)
                    label = str(int(node.action))

                circ = Circle(
                    (x, y),
                    leaf_r,
                    facecolor=face,
                    edgecolor=edge_color,
                    linewidth=1.9,
                    zorder=10,
                )
                ax.add_patch(circ)
                ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=LEAF_FS,
                    fontweight="bold",
                    zorder=10000,
                    clip_on=False,
                )

        draw(root)

        all_xy = list(pos.values())
        xs, ys = zip(*all_xy) if all_xy else ([0.0], [0.0])

        pad_x = 4.0 * node_scale
        pad_y = 4.5 * node_scale
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        ax.margins(x=0.10, y=0.14)

        for t in ax.texts:
            t.set_clip_on(False)

        ax.set_aspect("equal", adjustable="box")

        if title is not None:
            ax.set_title(title, fontsize=TITLE_FS, pad=18)

        if show:
            plt.show()

        return ax


class LicketySPLITRegressor(_BaseLicketySPLIT, RegressorMixin):
    def __init__(
        self,
        *,
        lambda_leaf: float = 0.01,
        depth_budget: int = 3,
        lookahead_k: int = 1,
        eta_defer=None,
        cache_mode="fingerprint",
        cost_caching_enabled: bool = True,
    ):
        self.lambda_leaf = lambda_leaf
        self.depth_budget = depth_budget
        self.lookahead_k = lookahead_k
        self.eta_defer = eta_defer
        self.cache_mode = cache_mode
        self.cost_caching_enabled = cost_caching_enabled

    def fit(
        self,
        X,
        y,
        *,
        bb_pred=None,
        sample_weight=None,
    ):
        X = _as_u8_X(X)
        n = int(X.shape[0])
        n_features = int(X.shape[1])

        y = _as_f64_vec(y, n, "y")

        weights_arg = None
        if sample_weight is not None:
            weights_arg = _as_f64_weights(sample_weight, n)

        if bb_pred is None:
            bb_arg = y.astype(np.float64, copy=True)
            eta_arg = _huge_eta_no_defer(n, n_features, weights_arg)
            self.uses_defer_ = False
        else:
            bb_arg = _as_f64_vec(bb_pred, n, "bb_pred")
            eta_arg = 0.0 if self.eta_defer is None else float(self.eta_defer)
            self.uses_defer_ = True

        self._model = self._new_core()

        obj = self._model.fit_regression(
            X,
            y,
            bb_arg,
            float(self.lambda_leaf),
            float(eta_arg),
            int(self.depth_budget),
            int(self.lookahead_k),
            weights_arg,
        )

        self.objective_ = float(obj)
        self.n_features_in_ = n_features
        self.eta_defer_used_ = float(eta_arg)

        return self

    def fit_return(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return float(self.objective_)

    def predict(self, X, *, bb_pred=None) -> np.ndarray:
        X = _as_u8_X(X)
        n = int(X.shape[0])

        if not hasattr(self, "_model"):
            raise RuntimeError("Model has not been fit.")

        if getattr(self, "uses_defer_", False):
            if bb_pred is None:
                raise ValueError("This model was fit with deferral; pass bb_pred to predict.")
            bb_arg = _as_f64_vec(bb_pred, n, "bb_pred")
        else:
            bb_arg = np.zeros(n, dtype=np.float64)

        return np.asarray(self._model.predict_regression(X, bb_arg), dtype=np.float64)

    def predict_placeholder(self, X, placeholder=np.nan) -> np.ndarray:
        X = _as_u8_X(X)
        return np.asarray(
            self._model.predict_regression_placeholder(X, float(placeholder)),
            dtype=np.float64,
        )

    def score(self, X, y, sample_weight=None):
        X = _as_u8_X(X)
        y = _as_f64_vec(y, int(X.shape[0]), "y")
        pred = self.predict(X)

        if sample_weight is None:
            denom = np.sum((y - np.mean(y)) ** 2)
            if denom <= 0.0:
                return 0.0
            return float(1.0 - np.sum((y - pred) ** 2) / denom)

        w = _as_f64_weights(sample_weight, int(X.shape[0]))
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            return 0.0

        y_bar = float(np.sum(w * y) / w_sum)
        denom = float(np.sum(w * (y - y_bar) ** 2))
        if denom <= 0.0:
            return 0.0

        return float(1.0 - np.sum(w * (y - pred) ** 2) / denom)

    def leaf_counts(self) -> dict:
        if not hasattr(self, "_model"):
            raise RuntimeError("Model has not been fit.")
        return dict(self._model.regression_leaf_counts_single_tree())

    def leaf_paths(self) -> list[list[int]]:
        if not hasattr(self, "_model"):
            raise RuntimeError("Model has not been fit.")
        return list(self._model.regression_leaf_paths_single_tree())

    def leaf_values(self) -> list[float]:
        if not hasattr(self, "_model"):
            raise RuntimeError("Model has not been fit.")
        return list(self._model.regression_leaf_values_single_tree())

    def leaf_actions(self) -> list[int]:
        if not hasattr(self, "_model"):
            raise RuntimeError("Model has not been fit.")
        return list(self._model.regression_leaf_actions_single_tree())

    def get_tree_paths(self):
        if not hasattr(self, "_model"):
            raise RuntimeError("Model has not been fit.")
        paths, values = self._model.regression_leaf_paths_and_values_single_tree()
        return paths, values

    def split_counts(self, X) -> np.ndarray:
        X = _as_u8_X(X)

        if not hasattr(self, "_model"):
            raise RuntimeError("Model has not been fit.")

        return np.asarray(
            self._model.regression_split_counts_single_tree(X),
            dtype=np.int32,
        )

    def plot_tree(
        self,
        feature_names=None,
        figsize=(14, 9),
        ax=None,
        title=None,
        show=True,
        value_color_low="#dbeafe",
        value_color_high="#fecaca",
        defer_color="#fef3c7",
        split_node_color="#e5e7eb",
        edge_color="black",
        node_scale=1.35,
        x_scale=5.8,
        y_scale=4.3,
        value_precision=3,
    ):
        paths, values = self.get_tree_paths()

        class Node:
            __slots__ = ("feature", "left", "right", "value")

            def __init__(self):
                self.feature = None
                self.left = None
                self.right = None
                self.value = None

        def build_tree_from_paths(paths, values):
            root = Node()
            for path, val in zip(paths, values):
                cur = root
                for signed_f in path:
                    f = abs(signed_f) - 1
                    go_left = signed_f > 0
                    if cur.feature is None:
                        cur.feature = f
                        cur.left = Node()
                        cur.right = Node()
                    cur = cur.left if go_left else cur.right
                cur.value = float(val)
            return root

        def collect_leaves_in_order(node, leaves):
            if node is None:
                return
            if node.value is not None or (node.left is None and node.right is None):
                leaves.append(node)
                return
            collect_leaves_in_order(node.left, leaves)
            collect_leaves_in_order(node.right, leaves)

        def assign_positions_tree(root):
            positions = {}
            leaves = []
            collect_leaves_in_order(root, leaves)
            if not leaves:
                leaves = [root]
            leaf_x = {leaf: i for i, leaf in enumerate(leaves)}

            def dfs(node, depth):
                if node is None:
                    return
                if node.value is not None or (node.left is None and node.right is None):
                    x = leaf_x[node]
                    positions[node] = (x, -depth)
                    return
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)
                x_left, _ = positions[node.left]
                x_right, _ = positions[node.right]
                positions[node] = (0.5 * (x_left + x_right), -depth)

            dfs(root, 0)
            return positions, len(leaves)

        def _shrink_segment(x1, y1, x2, y2, r1, r2):
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx * dx + dy * dy) ** 0.5
            if dist == 0:
                return x1, y1, x2, y2
            ux = dx / dist
            uy = dy / dist
            return (x1 + ux * r1, y1 + uy * r1, x2 - ux * r2, y2 - uy * r2)

        def _lerp_color(c1, c2, t):
            a = np.array(to_rgb(c1), dtype=float)
            b = np.array(to_rgb(c2), dtype=float)
            out = (1.0 - t) * a + t * b
            return tuple(out)

        finite_values = np.array(
            [v for v in values if np.isfinite(float(v))],
            dtype=float,
        )

        if finite_values.size:
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
        else:
            vmin = 0.0
            vmax = 1.0

        def _value_color(v):
            if not np.isfinite(v):
                return defer_color
            if vmax <= vmin:
                return value_color_low
            t = (float(v) - vmin) / (vmax - vmin)
            t = max(0.0, min(1.0, t))
            return _lerp_color(value_color_low, value_color_high, t)

        def _value_label(v):
            if not np.isfinite(v):
                return "defer"
            return f"{float(v):.{int(value_precision)}g}"

        if feature_names is None:
            enc = [abs(v) for p in paths for v in p]
            max_f = (max(enc) - 1) if enc else -1
            feature_names = [f"f{j}" for j in range(max_f + 1)]

        root = build_tree_from_paths(paths, values)
        pos_local, _ = assign_positions_tree(root)

        pos = {}
        xs = []
        ys = []
        for node, (x, y) in pos_local.items():
            xx = x * x_scale
            yy = y * y_scale
            pos[node] = (xx, yy)
            xs.append(xx)
            ys.append(yy)

        minx, maxx = min(xs), max(xs)
        x_center = 0.5 * (minx + maxx)
        for node, (xx, yy) in list(pos.items()):
            pos[node] = (xx - x_center, yy)

        if ax is None:
            all_nodes_xy = list(pos.values())
            xs = [p[0] for p in all_nodes_xy] if all_nodes_xy else [0.0]
            ys = [p[1] for p in all_nodes_xy] if all_nodes_xy else [0.0]
            total_w = (max(xs) - min(xs)) if xs else 10.0
            total_h = (max(ys) - min(ys)) if ys else 6.0
            fig_w = max(figsize[0], total_w / 2.0)
            fig_h = max(figsize[1], total_h / 1.45)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        else:
            fig = ax.figure

        ax.set_axis_off()

        internal_r = 0.62 * node_scale
        leaf_r = 0.76 * node_scale
        edge_lw = 2.8 * node_scale

        SPLIT_FS = int(17 * node_scale)
        LEAF_FS = int(16 * node_scale)
        TITLE_FS = int(20 * node_scale)
        SPLIT_OFFSET_PTS = 16

        def _feat_name(f: int) -> str:
            if feature_names is not None and f < len(feature_names):
                return str(feature_names[f])
            return f"f{f}"

        def draw(node):
            x, y = pos[node]

            if node.feature is not None:
                if node.left is not None:
                    x2, y2 = pos[node.left]
                    r_child = leaf_r if node.left.value is not None else internal_r
                    sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, internal_r, r_child)
                    ax.add_line(Line2D([sx, ex], [sy, ey], color=edge_color, linewidth=edge_lw, zorder=1))
                    draw(node.left)

                if node.right is not None:
                    x2, y2 = pos[node.right]
                    r_child = leaf_r if node.right.value is not None else internal_r
                    sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, internal_r, r_child)
                    ax.add_line(Line2D([sx, ex], [sy, ey], color=edge_color, linewidth=edge_lw, zorder=1))
                    draw(node.right)

            if node.value is None:
                circ = Circle(
                    (x, y),
                    internal_r,
                    facecolor=split_node_color,
                    edgecolor=edge_color,
                    linewidth=1.9,
                    zorder=10,
                )
                ax.add_patch(circ)

                if node.feature is not None:
                    name = _feat_name(int(node.feature))
                    ax.annotate(
                        name,
                        xy=(x, y),
                        xycoords="data",
                        xytext=(0, SPLIT_OFFSET_PTS),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=SPLIT_FS,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.98),
                        zorder=10000,
                        clip_on=False,
                    )
            else:
                v = float(node.value)
                face = _value_color(v)
                label = _value_label(v)

                circ = Circle(
                    (x, y),
                    leaf_r,
                    facecolor=face,
                    edgecolor=edge_color,
                    linewidth=1.9,
                    zorder=10,
                )
                ax.add_patch(circ)

                ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=LEAF_FS,
                    fontweight="bold",
                    zorder=10000,
                    clip_on=False,
                )

        draw(root)

        all_xy = list(pos.values())
        xs, ys = zip(*all_xy) if all_xy else ([0.0], [0.0])

        pad_x = 4.0 * node_scale
        pad_y = 4.5 * node_scale
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        ax.margins(x=0.10, y=0.14)

        for t in ax.texts:
            t.set_clip_on(False)

        ax.set_aspect("equal", adjustable="box")

        if title is not None:
            ax.set_title(title, fontsize=TITLE_FS, pad=18)

        if show:
            plt.show()

        return ax


LicketySPLIT = LicketySPLITClassifier