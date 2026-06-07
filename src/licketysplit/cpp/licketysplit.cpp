#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <memory>
#include <string>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <cstring>

using namespace std;

#if defined(_MSC_VER)
  #include <intrin.h>
  static inline int popcnt64(uint64_t x) { return static_cast<int>(__popcnt64(x)); }
  static inline int ctz64(uint64_t x) {
      unsigned long idx;
      _BitScanForward64(&idx, x);
      return static_cast<int>(idx);
  }
#else
  static inline int popcnt64(uint64_t x) { return __builtin_popcountll(x); }
  static inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
#endif

struct Packed {
    vector<uint64_t> w;
    Packed() = default;
    explicit Packed(size_t nwords) : w(nwords, 0ULL) {}

    inline bool any() const { for (uint64_t t : w) if (t) return true; return false; }
    inline int  count() const { int s = 0; for (uint64_t t : w) s += popcnt64(t); return s; }
};


static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

static inline uint64_t hash_mask64(const uint64_t* w, int n_words, uint64_t tail_mask) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < n_words; ++i) {
        uint64_t x = w[i];
        if (i == n_words - 1) x &= tail_mask;
        const uint64_t m = mix64(x);
        h ^= m + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

// K2: greedy cache key = (subproblem, depth)
struct K2HashKey {
    uint64_t k = 0;
    int depth = 0;

    bool operator==(const K2HashKey& o) const {
        return k == o.k && depth == o.depth;
    }

    struct Hash {
        size_t operator()(const K2HashKey& x) const noexcept {
            size_t h = static_cast<size_t>(x.k);
            const size_t d = static_cast<size_t>(x.depth);
            h ^= d + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
};

// K3: LicketySPLIT cost cache key = (subproblem, depth, lookahead k)
struct K3HashKey {
    uint64_t k = 0;
    int depth = 0;
    int lookahead = 0;

    bool operator==(const K3HashKey& o) const {
        return k == o.k && depth == o.depth && lookahead == o.lookahead;
    }

    struct Hash {
        size_t operator()(const K3HashKey& x) const noexcept {
            size_t h = static_cast<size_t>(x.k);
            const size_t d = static_cast<size_t>(x.depth);
            const size_t a = static_cast<size_t>(x.lookahead);
            h ^= d + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= a + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
};

struct K2BitKey {
    std::string bits;
    int depth = 0;

    bool operator==(const K2BitKey& o) const {
        return depth == o.depth && bits == o.bits;
    }

    struct Hash {
        size_t operator()(const K2BitKey& x) const noexcept {
            size_t h = std::hash<std::string>{}(x.bits);
            const size_t d = static_cast<size_t>(x.depth);
            h ^= d + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
};

struct K3BitKey {
    std::string bits;
    int depth = 0;
    int lookahead = 0;

    bool operator==(const K3BitKey& o) const {
        return depth == o.depth && lookahead == o.lookahead && bits == o.bits;
    }

    struct Hash {
        size_t operator()(const K3BitKey& x) const noexcept {
            size_t h = std::hash<std::string>{}(x.bits);
            const size_t d = static_cast<size_t>(x.depth);
            const size_t a = static_cast<size_t>(x.lookahead);
            h ^= d + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= a + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };
};

class LicketySPLIT {
public:
    enum class LeafKind : uint8_t { PREDICT = 0, DEFER = 1 };
    enum class CacheMode : uint8_t { HASH_FINGERPRINT = 0, BITVECTOR = 1 };

    void set_cache_mode(CacheMode mode) {
        cache_mode_ = mode;
        clear_cost_caches();
    }

    void set_cost_caching_enabled(bool enabled) {
        cost_caching_enabled_ = enabled;
        if (!enabled) clear_cost_caches();
    }

    void clear_cost_caches() const {
        greedy_hash_cache_.clear();
        lickety_hash_cache_.clear();
        greedy_bit_cache_.clear();
        lickety_bit_cache_.clear();
    }

    double last_objective() const { return last_objective_; }

    void fit(
        const vector<vector<uint8_t>>& X_row_major,
        const vector<int>& y,
        const vector<int>& bb_pred,
        double lambda_leaf,
        double eta_defer,
        int8_t depth_budget,
        int8_t lookahead_k = 1,
        const vector<double>& sample_weights = {}
    ) {
        if (X_row_major.empty()) throw runtime_error("fit: X has 0 rows.");
        if (X_row_major[0].empty()) throw runtime_error("fit: X has 0 features.");
        if (y.empty()) throw runtime_error("fit: y is empty.");
        if ((int)X_row_major.size() != (int)y.size()) throw runtime_error("fit: X and y size mismatch.");
        if ((int)bb_pred.size() != (int)y.size()) throw runtime_error("fit: bb_pred and y size mismatch.");
        if (lookahead_k < 1) throw runtime_error("fit: lookahead_k must be >= 1.");
        if ((int)X_row_major[0].size() <= 0) throw runtime_error("fit: invalid feature count.");
        if (!sample_weights.empty() && (int)sample_weights.size() != (int)y.size())
            throw runtime_error("fit: sample_weights and y size mismatch.");

        n_samples  = (int)y.size();
        n_features = (int)X_row_major[0].size();

        for (int i = 0; i < n_samples; ++i) {
            if ((int)X_row_major[(size_t)i].size() != n_features)
                throw runtime_error("fit: inconsistent row lengths in X.");
        }

        sample_weights_.assign((size_t)n_samples, 1.0);
        uniform_weights_fast_ = true;
        uniform_weight_ = 1.0;

        if (!sample_weights.empty()) {
            sample_weights_ = sample_weights;
            uniform_weight_ = sample_weights_[0];

            const double tol = 1e-12 * std::max(1.0, std::abs(uniform_weight_));
            for (double w : sample_weights_) {
                if (!std::isfinite(w)) throw runtime_error("fit: sample_weights contains non-finite value.");
                if (w < 0.0) throw runtime_error("fit: sample_weights contains negative value.");
                if (std::abs(w - uniform_weight_) > tol) {
                    uniform_weights_fast_ = false;
                }
            }
        }

        total_weight_ = 0.0;
        if (uniform_weights_fast_) {
            total_weight_ = uniform_weight_ * (double)n_samples;
        } else {
            for (double w : sample_weights_) {
                total_weight_ += w;
            }
        }

        n_words   = (n_samples + 63) / 64;
        tail_mask = (n_samples % 64) ? ((1ULL << (n_samples % 64)) - 1ULL) : ~0ULL;

        int max_y = -1;
        for (int yi : y) {
            if (yi < 0) throw runtime_error("fit: y labels must be nonnegative contiguous integers.");
            max_y = std::max(max_y, yi);
        }
        num_classes = max_y + 1;
        if (num_classes <= 0) throw runtime_error("fit: invalid number of classes.");

        vector<uint8_t> seen((size_t)num_classes, 0);
        for (int yi : y) seen[(size_t)yi] = 1;
        for (int c = 0; c < num_classes; ++c) {
            if (!seen[(size_t)c]) {
                throw runtime_error("fit: y labels must be contiguous 0,1,2,...,C-1.");
            }
        }

        for (int bi : bb_pred) {
            if (bi < 0 || bi >= num_classes) {
                throw runtime_error("fit: bb_pred values must be in the same class range as y.");
            }
        }

        lamN = lambda_leaf * total_weight_;
        eta = eta_defer;
        depth_trained = depth_budget;
        k_trained     = lookahead_k;

        X_bits.assign((size_t)n_features, Packed((size_t)n_words));
        for (int f = 0; f < n_features; ++f) {
            auto& col = X_bits[(size_t)f].w;
            for (int i = 0; i < n_samples; ++i) {
                if (X_row_major[(size_t)i][(size_t)f] & 1)
                    col[(size_t)(i >> 6)] |= (1ULL << (i & 63));
            }
            col[(size_t)(n_words - 1)] &= tail_mask;
        }

        Y_bits.assign((size_t)num_classes, Packed((size_t)n_words));
        for (int i = 0; i < n_samples; ++i) {
            const int yi = y[(size_t)i];
            Y_bits[(size_t)yi].w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
        }
        for (int c = 0; c < num_classes; ++c) {
            Y_bits[(size_t)c].w[(size_t)(n_words - 1)] &= tail_mask;
        }

        // special cases to speed up binary classification
        Y1 = (num_classes > 1) ? Y_bits[(size_t)1] : Packed((size_t)n_words);

        BBwrong = Packed((size_t)n_words);
        for (int i = 0; i < n_samples; ++i) {
            const int yi = y[(size_t)i];
            const int bi = bb_pred[(size_t)i];
            if (yi != bi) BBwrong.w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
        }
        BBwrong.w[(size_t)(n_words - 1)] &= tail_mask;

        clear_cost_caches();

        Packed root((size_t)n_words);
        for (int w = 0; w < n_words - 1; ++w) root.w[(size_t)w] = ~0ULL;
        root.w[(size_t)(n_words - 1)] = tail_mask;

        double obj = 0.0;
        root_node = build_lickety_tree(root, depth_budget, lookahead_k, obj);
        last_objective_ = obj;
    }

    vector<int> predict(
        const vector<vector<uint8_t>>& X_row_major,
        const vector<int>& bb_pred_row = {},
        int placeholder = 99
    ) const {
        if (!root_node) throw runtime_error("predict: Model not fit.");
        const size_t N = X_row_major.size();
        if (N == 0) return {};
        if ((int)X_row_major[0].size() != n_features) throw runtime_error("predict: feature count mismatch.");

        const bool use_placeholder = bb_pred_row.empty();
        if (!use_placeholder && bb_pred_row.size() != N) throw runtime_error("predict: bb_pred_row size mismatch.");

        const vector<int> dummy_bb(use_placeholder ? N : 0, 0);
        const vector<int>& bb = use_placeholder ? dummy_bb : bb_pred_row;

        vector<int> out(N, 0);

        vector<int> idx; idx.reserve(N);
        for (size_t i = 0; i < N; ++i) idx.push_back((int)i);

        predict_rec(root_node.get(), X_row_major, bb, out, idx, use_placeholder, placeholder);
        return out;
    }

    struct LeafCounts {
        std::vector<int> predict_by_class;
        int defer = 0;

        int total() const {
            int s = defer;
            for (int v : predict_by_class) s += v;
            return s;
        }
    };

    LeafCounts leaf_counts_single_tree() const {
        LeafCounts c;
        c.predict_by_class.assign((size_t)num_classes, 0);
        if (!root_node) return c;
        count_leaves_rec_(root_node.get(), c);
        return c;
    }

    vector<int> split_counts_single_tree(
        const vector<vector<uint8_t>>& X_row_major
    ) const {
        if (!root_node) throw runtime_error("split_counts_single_tree: Model not fit.");
        const size_t N = X_row_major.size();
        if (N == 0) return {};
        if ((int)X_row_major[0].size() != n_features)
            throw runtime_error("split_counts_single_tree: feature count mismatch.");

        vector<int> out(N, 0);
        for (size_t i = 0; i < N; ++i) {
            out[i] = split_count_one_(root_node.get(), X_row_major[i]);
        }
        return out;
    }

    std::vector<std::vector<int>> leaf_paths_single_tree() const {
        std::vector<std::vector<int>> paths;
        if (!root_node) return paths;
        std::vector<int> cur;
        collect_leaf_paths_rec_(root_node.get(), cur, paths);
        return paths;
    }

    std::vector<int> leaf_actions_single_tree() const {
        std::vector<int> actions;
        if (!root_node) return actions;
        std::vector<int> cur;
        std::vector<std::vector<int>> dummy_paths;
        collect_leaf_paths_and_actions_rec_(root_node.get(), cur, dummy_paths, actions);
        return actions;
    }

    std::pair<std::vector<std::vector<int>>, std::vector<int>> leaf_paths_and_actions_single_tree() const {
        std::vector<std::vector<int>> paths;
        std::vector<int> actions;
        if (!root_node) return {paths, actions};
        std::vector<int> cur;
        collect_leaf_paths_and_actions_rec_(root_node.get(), cur, paths, actions);
        return {paths, actions};
    }

private:
    struct Node {
        int feature = -1;
        LeafKind kind = LeafKind::PREDICT;
        int pred_class = 0;
        shared_ptr<Node> left;
        shared_ptr<Node> right;
    };

    int n_samples = 0;
    int n_features = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL;

    double lamN = 0.0;
    double eta = 0.0;
    int8_t depth_trained = -1;
    int8_t k_trained = 1;

    vector<double> sample_weights_;
    double total_weight_ = 0.0;
    bool uniform_weights_fast_ = true;
    double uniform_weight_ = 1.0;

    vector<Packed> X_bits;
    int num_classes = 0;
    vector<Packed> Y_bits;
    Packed Y1;
    Packed BBwrong;

    shared_ptr<Node> root_node;
    double last_objective_ = 0.0;

    CacheMode cache_mode_ = CacheMode::HASH_FINGERPRINT;
    bool cost_caching_enabled_ = true;

    mutable unordered_map<K2HashKey, double, K2HashKey::Hash> greedy_hash_cache_;
    mutable unordered_map<K3HashKey, double, K3HashKey::Hash> lickety_hash_cache_;
    mutable unordered_map<K2BitKey, double, K2BitKey::Hash> greedy_bit_cache_;
    mutable unordered_map<K3BitKey, double, K3BitKey::Hash> lickety_bit_cache_;

    inline uint64_t key_of_mask_hash_(const Packed& mask) const {
        return hash_mask64(mask.w.data(), n_words, tail_mask);
    }

    inline std::string key_of_mask_bits_(const Packed& mask) const {
        const size_t bytes = (size_t)n_words * sizeof(uint64_t);
        std::string key;
        key.resize(bytes);
        for (int i = 0; i < n_words; ++i) {
            uint64_t x = mask.w[(size_t)i];
            if (i == n_words - 1) x &= tail_mask;
            std::memcpy(&key[(size_t)i * sizeof(uint64_t)], &x, sizeof(uint64_t));
        }
        return key;
    }

    inline bool try_get_greedy_cached_(const Packed& mask, int8_t depth, double& out) const {
        if (!cost_caching_enabled_) return false;

        if (cache_mode_ == CacheMode::HASH_FINGERPRINT) {
            const K2HashKey key{key_of_mask_hash_(mask), (int)depth};
            auto it = greedy_hash_cache_.find(key);
            if (it == greedy_hash_cache_.end()) return false;
            out = it->second;
            return true;
        }

        const K2BitKey key{key_of_mask_bits_(mask), (int)depth};
        auto it = greedy_bit_cache_.find(key);
        if (it == greedy_bit_cache_.end()) return false;
        out = it->second;
        return true;
    }

    inline void put_greedy_cached_(const Packed& mask, int8_t depth, double val) const {
        if (!cost_caching_enabled_) return;

        if (cache_mode_ == CacheMode::HASH_FINGERPRINT) {
            greedy_hash_cache_.emplace(K2HashKey{key_of_mask_hash_(mask), (int)depth}, val);
        } else {
            greedy_bit_cache_.emplace(K2BitKey{key_of_mask_bits_(mask), (int)depth}, val);
        }
    }

    inline bool try_get_lickety_cached_(const Packed& mask, int8_t depth, int8_t k, double& out) const {
        if (!cost_caching_enabled_) return false;

        if (cache_mode_ == CacheMode::HASH_FINGERPRINT) {
            const K3HashKey key{key_of_mask_hash_(mask), (int)depth, (int)k};
            auto it = lickety_hash_cache_.find(key);
            if (it == lickety_hash_cache_.end()) return false;
            out = it->second;
            return true;
        }

        const K3BitKey key{key_of_mask_bits_(mask), (int)depth, (int)k};
        auto it = lickety_bit_cache_.find(key);
        if (it == lickety_bit_cache_.end()) return false;
        out = it->second;
        return true;
    }

    inline void put_lickety_cached_(const Packed& mask, int8_t depth, int8_t k, double val) const {
        if (!cost_caching_enabled_) return;

        if (cache_mode_ == CacheMode::HASH_FINGERPRINT) {
            lickety_hash_cache_.emplace(K3HashKey{key_of_mask_hash_(mask), (int)depth, (int)k}, val);
        } else {
            lickety_bit_cache_.emplace(K3BitKey{key_of_mask_bits_(mask), (int)depth, (int)k}, val);
        }
    }

    inline void and_bits(const Packed& a, const Packed& b, Packed& out) const {
        for (int i = 0; i < n_words; ++i) out.w[(size_t)i] = a.w[(size_t)i] & b.w[(size_t)i];
        out.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    inline void andnot_bits(const Packed& a, const Packed& b, Packed& out) const {
        for (int i = 0; i < n_words; ++i) out.w[(size_t)i] = a.w[(size_t)i] & ~b.w[(size_t)i];
        out.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    inline void split_bits_count_left(
        const Packed& mask,
        const Packed& Xf,
        Packed& L,
        Packed& R,
        int& left_n
    ) const {
        left_n = 0;
        for (int i = 0; i < n_words; ++i) {
            const uint64_t mw = mask.w[(size_t)i];
            const uint64_t xw = Xf.w[(size_t)i];
            const uint64_t lw = mw & xw;
            L.w[(size_t)i] = lw;
            R.w[(size_t)i] = mw & ~xw;
            left_n += popcnt64(lw);
        }
        L.w[(size_t)(n_words - 1)] &= tail_mask;
        R.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    inline bool cheap_leaf_done_(double leaf_cost) const {
        return leaf_cost <= 2.0 * lamN;
    }

    inline int count_and(const Packed& a, const Packed& b) const {
        int s = 0;
        for (int i = 0; i < n_words; ++i) {
            s += popcnt64(a.w[(size_t)i] & b.w[(size_t)i]);
        }
        return s;
    }

    inline int count_and3(const Packed& a, const Packed& b, const Packed& c) const {
        int s = 0;
        for (int i = 0; i < n_words; ++i) {
            s += popcnt64(a.w[(size_t)i] & b.w[(size_t)i] & c.w[(size_t)i]);
        }
        return s;
    }

    inline double mass_from_count(int count) const {
        return uniform_weight_ * (double)count;
    }

    inline double weight_sum_mask(const Packed& mask) const {
        if (uniform_weights_fast_) {
            return mass_from_count(mask.count());
        }

        double s = 0.0;
        for (int wi = 0; wi < n_words; ++wi) {
            uint64_t bits = mask.w[(size_t)wi];
            while (bits) {
                const int b = ctz64(bits);
                const int idx = (wi << 6) + b;
                s += sample_weights_[(size_t)idx];
                bits &= (bits - 1ULL);
            }
        }
        return s;
    }

    inline double weight_sum_and(const Packed& a, const Packed& b) const {
        if (uniform_weights_fast_) {
            return mass_from_count(count_and(a, b));
        }

        double s = 0.0;
        for (int wi = 0; wi < n_words; ++wi) {
            uint64_t bits = a.w[(size_t)wi] & b.w[(size_t)wi];
            while (bits) {
                const int bit = ctz64(bits);
                const int idx = (wi << 6) + bit;
                s += sample_weights_[(size_t)idx];
                bits &= (bits - 1ULL);
            }
        }
        return s;
    }

    static inline double entropy_bin(double pos_mass, double total_mass) {
        if (total_mass <= 0.0) return 0.0;
        const double eps = 1e-12;
        double p = pos_mass / total_mass;
        p = max(eps, min(1.0 - eps, p));
        return -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
    }

    static inline double entropy_multiclass(const std::vector<double>& class_masses, double total_mass) {
        if (total_mass <= 0.0) return 0.0;
        double H = 0.0;
        for (double m : class_masses) {
            if (m <= 0.0) continue;
            const double p = m / total_mass;
            H -= p * log2(p);
        }
        return H;
    }

    inline void class_masses_mask(const Packed& mask, std::vector<double>& masses) const {
        masses.assign((size_t)num_classes, 0.0);

        if (uniform_weights_fast_) {
            for (int c = 0; c < num_classes; ++c) {
                masses[(size_t)c] = mass_from_count(count_and(mask, Y_bits[(size_t)c]));
            }
            return;
        }

        for (int c = 0; c < num_classes; ++c) {
            masses[(size_t)c] = weight_sum_and(mask, Y_bits[(size_t)c]);
        }
    }

    struct BestLeaf {
        double cost = std::numeric_limits<double>::infinity();
        LeafKind kind = LeafKind::PREDICT;
        int pred_class = 0;
    };

    BestLeaf best_leaf_option_uniform(const Packed& mask) const {
        const int n = mask.count();
        const double mass = mass_from_count(n);
        if (mass <= 0.0) return BestLeaf{0.0, LeafKind::PREDICT, 0};

        BestLeaf best;

        if (num_classes == 2) {
            const int pos = count_and(mask, Y1);
            const int neg = n - pos;

            const double cost0 = lamN + mass_from_count(pos);
            const double cost1 = lamN + mass_from_count(neg);

            best = BestLeaf{cost0, LeafKind::PREDICT, 0};
            if (cost1 < best.cost) best = BestLeaf{cost1, LeafKind::PREDICT, 1};
        } else {
            best = BestLeaf{std::numeric_limits<double>::infinity(), LeafKind::PREDICT, 0};

            for (int c = 0; c < num_classes; ++c) {
                const int class_count = count_and(mask, Y_bits[(size_t)c]);
                const double cost = lamN + mass_from_count(n - class_count);

                if (cost < best.cost) {
                    best = BestLeaf{cost, LeafKind::PREDICT, c};
                }
            }
        }

        const double bb_mis_mass = mass_from_count(count_and(mask, BBwrong));
        const double costD = lamN + eta * mass + bb_mis_mass;
        if (costD < best.cost) best = BestLeaf{costD, LeafKind::DEFER, 0};

        return best;
    }

    BestLeaf best_leaf_option_weighted(const Packed& mask) const {
        const double mass = weight_sum_mask(mask);
        if (mass <= 0.0) return BestLeaf{0.0, LeafKind::PREDICT, 0};

        BestLeaf best;

        if (num_classes == 2) {
            // binary case: only count class 1; class 0 is the remainder.
            const double pos_mass = weight_sum_and(mask, Y1);
            const double neg_mass = mass - pos_mass;

            const double cost0 = lamN + pos_mass;
            const double cost1 = lamN + neg_mass;

            best = BestLeaf{cost0, LeafKind::PREDICT, 0};
            if (cost1 < best.cost) best = BestLeaf{cost1, LeafKind::PREDICT, 1};
        } else {
            best = BestLeaf{std::numeric_limits<double>::infinity(), LeafKind::PREDICT, 0};

            for (int c = 0; c < num_classes; ++c) {
                const double class_mass = weight_sum_and(mask, Y_bits[(size_t)c]);
                const double cost = lamN + (mass - class_mass);

                if (cost < best.cost) {
                    best = BestLeaf{cost, LeafKind::PREDICT, c};
                }
            }
        }

        const double bb_mis_mass = weight_sum_and(mask, BBwrong);
        const double defer_pen = eta * mass;
        const double costD = lamN + defer_pen + bb_mis_mass;
        if (costD < best.cost) best = BestLeaf{costD, LeafKind::DEFER, 0};

        return best;
    }

    BestLeaf best_leaf_option(const Packed& mask) const {
        if (uniform_weights_fast_) {
            return best_leaf_option_uniform(mask);
        }
        return best_leaf_option_weighted(mask);
    }

    // exact depth 1 solver: uses greedy cache, no depth 0 caches
    double depth1_optimal_cost_binary_uniform_nocache(const Packed& mask) const {
        const int n_sub = mask.count();
        if (n_sub <= 0) return 0.0;

        const int pos_total = count_and(mask, Y1);
        const int bb_wrong_total = count_and(mask, BBwrong);

        double ans = lamN + mass_from_count(std::min(pos_total, n_sub - pos_total));

        const double costD = lamN
            + eta * mass_from_count(n_sub)
            + mass_from_count(bb_wrong_total);
        if (costD < ans) ans = costD;

        if (cheap_leaf_done_(ans)) return ans;
        if (n_sub <= 1) return ans;

        for (int f = 0; f < n_features; ++f) {
            const Packed& Xf = X_bits[(size_t)f];

            int left_n = 0;
            int left_pos = 0;
            int left_bb_wrong = 0;

            for (int i = 0; i < n_words; ++i) {
                const uint64_t lw = mask.w[(size_t)i] & Xf.w[(size_t)i];
                left_n += popcnt64(lw);
                left_pos += popcnt64(lw & Y1.w[(size_t)i]);
                left_bb_wrong += popcnt64(lw & BBwrong.w[(size_t)i]);
            }

            const int right_n = n_sub - left_n;
            if (left_n == 0 || right_n == 0) continue;

            const int right_pos = pos_total - left_pos;
            const int right_bb_wrong = bb_wrong_total - left_bb_wrong;

            double left_leaf = lamN + mass_from_count(std::min(left_pos, left_n - left_pos));
            const double left_defer = lamN
                + eta * mass_from_count(left_n)
                + mass_from_count(left_bb_wrong);
            if (left_defer < left_leaf) left_leaf = left_defer;

            double right_leaf = lamN + mass_from_count(std::min(right_pos, right_n - right_pos));
            const double right_defer = lamN
                + eta * mass_from_count(right_n)
                + mass_from_count(right_bb_wrong);
            if (right_defer < right_leaf) right_leaf = right_defer;

            const double split_cost = left_leaf + right_leaf;
            if (split_cost < ans) ans = split_cost;
        }

        return ans;
    }

    double depth1_optimal_cost(const Packed& mask) const {
        constexpr int8_t DEPTH = 1;

        double cached = 0.0;
        if (try_get_greedy_cached_(mask, DEPTH, cached)) return cached;

        double ans = 0.0;

        if (uniform_weights_fast_ && num_classes == 2) {
            ans = depth1_optimal_cost_binary_uniform_nocache(mask);
        } else {
            BestLeaf leaf = best_leaf_option(mask);
            ans = leaf.cost;

            if (cheap_leaf_done_(ans)) {
                put_greedy_cached_(mask, DEPTH, ans);
                return ans;
            }

            const int D_count = mask.count();
            if (D_count > 1) {
                Packed L((size_t)n_words), R((size_t)n_words);

                for (int f = 0; f < n_features; ++f) {
                    int left_n = 0;
                    split_bits_count_left(mask, X_bits[(size_t)f], L, R, left_n);
                    if (left_n == 0 || left_n == D_count) continue;

                    const double split_cost =
                        best_leaf_option(L).cost
                        + best_leaf_option(R).cost;

                    if (split_cost < ans) ans = split_cost;
                }
            }
        }

        put_greedy_cached_(mask, DEPTH, ans);
        return ans;
    }

    double greedy_cost(const Packed& mask, int8_t depth) const {
        // do not cache depth-0 leaf results.
        if (depth <= 0) {
            return best_leaf_option(mask).cost;
        }

        // depth 1 is solved exactly and cached in the greedy K2 cache.
        if (depth == 1) {
            return depth1_optimal_cost(mask);
        }

        double cached = 0.0;
        if (try_get_greedy_cached_(mask, depth, cached)) return cached;

        BestLeaf leaf = best_leaf_option(mask);
        double ans = leaf.cost;

        if (cheap_leaf_done_(ans)) {
            put_greedy_cached_(mask, depth, ans);
            return ans;
        }

        const int D_count = mask.count();
        if (D_count > 1) {
            const int best_f = find_best_split_entropy(mask);

            if (best_f >= 0) {
                Packed L((size_t)n_words), R((size_t)n_words);
                int left_n = 0;
                split_bits_count_left(mask, X_bits[(size_t)best_f], L, R, left_n);

                if (left_n > 0 && left_n < D_count) {
                    const double split_cost =
                        greedy_cost(L, (int8_t)(depth - 1))
                        + greedy_cost(R, (int8_t)(depth - 1));

                    ans = std::min(leaf.cost, split_cost);
                }
            }
        }

        put_greedy_cached_(mask, depth, ans);
        return ans;
    }

    double lickety_cost(const Packed& mask, int8_t depth, int8_t k) const {
        // do not cache depth-0 leaf results.
        if (depth <= 0) {
            return best_leaf_option(mask).cost;
        }

        // clamp lookahead to the the smallest optimal value if already optimal
        if (k > depth - 1) {
            k = (int8_t)(depth - 1);
        }

        // optimal depth 1 solver - reuse greedy optimal depth 1 cache
        if (depth == 1) {
            return depth1_optimal_cost(mask);
        }

        double cached = 0.0;
        if (try_get_lickety_cached_(mask, depth, k, cached)) return cached;

        BestLeaf leaf = best_leaf_option(mask);
        double ans = leaf.cost;

        if (cheap_leaf_done_(ans)) {
            put_lickety_cached_(mask, depth, k, ans);
            return ans;
        }

        const int D_count = mask.count();
        if (D_count > 1) {
            int best_feat = -1;
            double best_proxy_children = std::numeric_limits<double>::infinity();

            Packed L((size_t)n_words), R((size_t)n_words);

            for (int f = 0; f < n_features; ++f) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[(size_t)f], L, R, left_n);
                if (left_n == 0 || left_n == D_count) continue;

                double proxy_children = 0.0;
                if (k <= 1) {
                    proxy_children =
                        greedy_cost(L, (int8_t)(depth - 1))
                        + greedy_cost(R, (int8_t)(depth - 1));
                } else {
                    const int8_t km1 = (int8_t)(k - 1);
                    proxy_children =
                        lickety_cost(L, (int8_t)(depth - 1), km1)
                        + lickety_cost(R, (int8_t)(depth - 1), km1);
                }

                if (proxy_children < best_proxy_children) {
                    best_proxy_children = proxy_children;
                    best_feat = f;
                }
            }

            if (best_feat >= 0) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[(size_t)best_feat], L, R, left_n);

                if (left_n > 0 && left_n < D_count) {
                    const double split_cost =
                        lickety_cost(L, (int8_t)(depth - 1), k)
                        + lickety_cost(R, (int8_t)(depth - 1), k);

                    ans = std::min(leaf.cost, split_cost);
                }
            }
        }

        put_lickety_cached_(mask, depth, k, ans);
        return ans;
    }

    shared_ptr<Node> build_depth1_optimal_tree(
        const Packed& mask,
        double& out_cost
    ) const {
        BestLeaf leaf = best_leaf_option(mask);

        double best_cost = leaf.cost;
        if (cheap_leaf_done_(best_cost)) {
            out_cost = best_cost;
            return make_leaf_(leaf);
        }

        int best_feat = -1;
        BestLeaf best_left_leaf;
        BestLeaf best_right_leaf;

        const int D_count = mask.count();
        if (D_count > 1) {
            Packed L((size_t)n_words), R((size_t)n_words);

            for (int f = 0; f < n_features; ++f) {
                int left_n = 0;
                split_bits_count_left(mask, X_bits[(size_t)f], L, R, left_n);
                if (left_n == 0 || left_n == D_count) continue;

                BestLeaf left_leaf = best_leaf_option(L);
                BestLeaf right_leaf = best_leaf_option(R);
                const double split_cost = left_leaf.cost + right_leaf.cost;

                if (split_cost < best_cost) {
                    best_cost = split_cost;
                    best_feat = f;
                    best_left_leaf = left_leaf;
                    best_right_leaf = right_leaf;
                }
            }
        }

        if (best_feat < 0) {
            out_cost = leaf.cost;
            return make_leaf_(leaf);
        }

        out_cost = best_cost;
        auto n = make_shared<Node>();
        n->feature = best_feat;
        n->left = make_leaf_(best_left_leaf);
        n->right = make_leaf_(best_right_leaf);
        return n;
    }

    shared_ptr<Node> build_lickety_tree(const Packed& mask, int8_t depth, int8_t k, double& out_cost) const {
        BestLeaf leaf = best_leaf_option(mask);

        if (depth <= 0) {
            out_cost = leaf.cost;
            return make_leaf_(leaf);
        }

        if (cheap_leaf_done_(leaf.cost)) {
            out_cost = leaf.cost;
            return make_leaf_(leaf);
        }

        if (depth == 1) {
            // make the final tree agree with the exact cached depth-1 objective.
            return build_depth1_optimal_tree(mask, out_cost);
        }

        const int D_count = mask.count();
        if (D_count <= 1) { out_cost = leaf.cost; return make_leaf_(leaf); }

        int best_feat = -1;
        double best_proxy_children = std::numeric_limits<double>::infinity();

        Packed bestL((size_t)n_words), bestR((size_t)n_words);
        Packed L((size_t)n_words), R((size_t)n_words);

        for (int f = 0; f < n_features; ++f) {
            int left_n = 0;
            split_bits_count_left(mask, X_bits[(size_t)f], L, R, left_n);
            if (left_n == 0 || left_n == D_count) continue;

            double proxy_children = 0.0;
            if (k <= 1) {
                proxy_children =
                    greedy_cost(L, (int8_t)(depth - 1))
                    + greedy_cost(R, (int8_t)(depth - 1));
            } else {
                const int8_t km1 = (int8_t)(k - 1);
                proxy_children =
                    lickety_cost(L, (int8_t)(depth - 1), km1)
                    + lickety_cost(R, (int8_t)(depth - 1), km1);
            }

            if (proxy_children < best_proxy_children) {
                best_proxy_children = proxy_children;
                best_feat = f;
                bestL.w = L.w;
                bestR.w = R.w;
            }
        }

        if (best_feat < 0) { out_cost = leaf.cost; return make_leaf_(leaf); }

        double left_cost = 0.0, right_cost = 0.0;
        auto left_node  = build_lickety_tree(bestL, (int8_t)(depth - 1), k, left_cost);
        auto right_node = build_lickety_tree(bestR, (int8_t)(depth - 1), k, right_cost);

        const double split_cost = left_cost + right_cost;

        if (leaf.cost <= split_cost) { out_cost = leaf.cost; return make_leaf_(leaf); }

        out_cost = split_cost;
        auto n = make_shared<Node>();
        n->feature = best_feat;
        n->left = left_node;
        n->right = right_node;
        return n;
    }

    int find_best_split_entropy(const Packed& mask) const {
        if (num_classes == 2) return find_best_split_entropy_binary(mask);
        return find_best_split_entropy_multiclass(mask);
    }

    int find_best_split_entropy_binary(const Packed& mask) const {
        const int D_count = mask.count();
        if (D_count <= 1) return -1;

        if (uniform_weights_fast_) {
            const double D_mass = mass_from_count(D_count);
            if (D_mass <= 0.0) return -1;

            const int pos_total = count_and(mask, Y1);
            const double pos_mass = mass_from_count(pos_total);
            const double baseH = entropy_bin(pos_mass, D_mass);

            int best_f = -1;
            double best_gain = -1e300;

            for (int f = 0; f < n_features; ++f) {
                const Packed& Xf = X_bits[(size_t)f];

                int left_n = 0;
                int left_pos = 0;

                for (int i = 0; i < n_words; ++i) {
                    const uint64_t lw = mask.w[(size_t)i] & Xf.w[(size_t)i];
                    left_n += popcnt64(lw);
                    left_pos += popcnt64(lw & Y1.w[(size_t)i]);
                }

                const int right_n = D_count - left_n;
                if (left_n == 0 || right_n == 0) continue;

                const int right_pos = pos_total - left_pos;

                const double Dl_mass = mass_from_count(left_n);
                const double Dr_mass = mass_from_count(right_n);
                if (Dl_mass <= 0.0 || Dr_mass <= 0.0) continue;

                const double wl = Dl_mass / D_mass;
                const double wr = Dr_mass / D_mass;

                const double Hl = entropy_bin(mass_from_count(left_pos), Dl_mass);
                const double Hr = entropy_bin(mass_from_count(right_pos), Dr_mass);

                const double gain = baseH - (wl * Hl + wr * Hr);
                if (gain > best_gain) { best_gain = gain; best_f = f; }
            }

            return best_f;
        }

        const double D_mass = weight_sum_mask(mask);
        if (D_mass <= 0.0) return -1;

        const double pos_mass = weight_sum_and(mask, Y1);
        const double baseH = entropy_bin(pos_mass, D_mass);

        int best_f = -1;
        double best_gain = -1e300;

        Packed L((size_t)n_words);
        for (int f = 0; f < n_features; ++f) {
            for (int i = 0; i < n_words; ++i) {
                L.w[(size_t)i] = mask.w[(size_t)i] & X_bits[(size_t)f].w[(size_t)i];
            }
            L.w[(size_t)(n_words - 1)] &= tail_mask;

            const int Dl_count = L.count();
            const int Dr_count = D_count - Dl_count;
            if (Dl_count == 0 || Dr_count == 0) continue;

            const double Dl_mass = weight_sum_mask(L);
            const double Dr_mass = D_mass - Dl_mass;
            if (Dl_mass <= 0.0 || Dr_mass <= 0.0) continue;

            const double posl_mass = weight_sum_and(L, Y1);
            const double posr_mass = pos_mass - posl_mass;

            const double wl = Dl_mass / D_mass;
            const double wr = Dr_mass / D_mass;

            const double Hl = entropy_bin(posl_mass, Dl_mass);
            const double Hr = entropy_bin(posr_mass, Dr_mass);

            const double gain = baseH - (wl * Hl + wr * Hr);
            if (gain > best_gain) { best_gain = gain; best_f = f; }
        }
        return best_f;
    }

    int find_best_split_entropy_multiclass(const Packed& mask) const {
        const int D_count = mask.count();
        if (D_count <= 1) return -1;

        const double D_mass = weight_sum_mask(mask);
        if (D_mass <= 0.0) return -1;

        std::vector<double> parent_masses;
        class_masses_mask(mask, parent_masses);
        const double baseH = entropy_multiclass(parent_masses, D_mass);

        int best_f = -1;
        double best_gain = -1e300;

        Packed L((size_t)n_words);
        std::vector<double> left_masses((size_t)num_classes, 0.0);
        std::vector<double> right_masses((size_t)num_classes, 0.0);

        for (int f = 0; f < n_features; ++f) {
            for (int i = 0; i < n_words; ++i) {
                L.w[(size_t)i] = mask.w[(size_t)i] & X_bits[(size_t)f].w[(size_t)i];
            }
            L.w[(size_t)(n_words - 1)] &= tail_mask;

            const int Dl_count = L.count();
            const int Dr_count = D_count - Dl_count;
            if (Dl_count == 0 || Dr_count == 0) continue;

            const double Dl_mass = weight_sum_mask(L);
            const double Dr_mass = D_mass - Dl_mass;
            if (Dl_mass <= 0.0 || Dr_mass <= 0.0) continue;

            class_masses_mask(L, left_masses);
            for (int c = 0; c < num_classes; ++c) {
                right_masses[(size_t)c] = parent_masses[(size_t)c] - left_masses[(size_t)c];
            }

            const double wl = Dl_mass / D_mass;
            const double wr = Dr_mass / D_mass;

            const double Hl = entropy_multiclass(left_masses, Dl_mass);
            const double Hr = entropy_multiclass(right_masses, Dr_mass);

            const double gain = baseH - (wl * Hl + wr * Hr);
            if (gain > best_gain) { best_gain = gain; best_f = f; }
        }

        return best_f;
    }

    static shared_ptr<Node> make_leaf_(const BestLeaf& leaf) {
        auto n = make_shared<Node>();
        n->feature = -1;
        n->kind = leaf.kind;
        n->pred_class = leaf.pred_class;
        return n;
    }

    static void predict_rec(
        const Node* node,
        const vector<vector<uint8_t>>& X_row_major,
        const vector<int>& bb_pred_row,
        vector<int>& out,
        const vector<int>& idx,
        bool use_placeholder,
        int placeholder
    ) {
        if (!node) return;

        if (node->feature < 0) {
            if (node->kind == LeafKind::DEFER) {
                if (use_placeholder) {
                    for (int r : idx) out[(size_t)r] = placeholder;
                } else {
                    for (int r : idx) out[(size_t)r] = bb_pred_row[(size_t)r];
                }
            } else {
                const int p = node->pred_class;
                for (int r : idx) out[(size_t)r] = p;
            }
            return;
        }

        const int f = node->feature;
        vector<int> left_idx;  left_idx.reserve(idx.size());
        vector<int> right_idx; right_idx.reserve(idx.size());

        for (int r : idx) {
            if (X_row_major[(size_t)r][(size_t)f] & 1) left_idx.push_back(r);
            else right_idx.push_back(r);
        }

        if (!left_idx.empty())  predict_rec(node->left.get(),  X_row_major, bb_pred_row, out, left_idx, use_placeholder, placeholder);
        if (!right_idx.empty()) predict_rec(node->right.get(), X_row_major, bb_pred_row, out, right_idx, use_placeholder, placeholder);
    }

    void count_leaves_rec_(const Node* node, LeafCounts& c) const {
        if (!node) return;

        if (node->feature < 0) {
            if (node->kind == LeafKind::DEFER) {
                c.defer += 1;
            } else {
                const int p = node->pred_class;
                if (p >= 0 && p < (int)c.predict_by_class.size()) {
                    c.predict_by_class[(size_t)p] += 1;
                }
            }
            return;
        }

        if (node->left)  count_leaves_rec_(node->left.get(), c);
        if (node->right) count_leaves_rec_(node->right.get(), c);
    }

    static int split_count_one_(const Node* node, const vector<uint8_t>& xrow) {
        int s = 0;
        const Node* cur = node;
        while (cur && cur->feature >= 0) {
            const int f = cur->feature;
            ++s;
            cur = (xrow[(size_t)f] & 1) ? cur->left.get() : cur->right.get();
        }
        return s;
    }

    static void collect_leaf_paths_rec_(
        const Node* node,
        std::vector<int>& cur,
        std::vector<std::vector<int>>& out
    ) {
        if (!node) return;

        if (node->feature < 0) {
            out.push_back(cur);
            return;
        }

        const int f1 = node->feature + 1;

        cur.push_back(+f1);
        collect_leaf_paths_rec_(node->left.get(), cur, out);
        cur.pop_back();

        cur.push_back(-f1);
        collect_leaf_paths_rec_(node->right.get(), cur, out);
        cur.pop_back();
    }

    static void collect_leaf_paths_and_actions_rec_(
        const Node* node,
        std::vector<int>& cur,
        std::vector<std::vector<int>>& paths_out,
        std::vector<int>& actions_out
    ) {
        if (!node) return;

        if (node->feature < 0) {
            paths_out.push_back(cur);
            if (node->kind == LeafKind::DEFER) {
                actions_out.push_back(-1);
            } else {
                actions_out.push_back(node->pred_class);
            }
            return;
        }

        const int f1 = node->feature + 1;

        cur.push_back(+f1);
        collect_leaf_paths_and_actions_rec_(node->left.get(), cur, paths_out, actions_out);
        cur.pop_back();

        cur.push_back(-f1);
        collect_leaf_paths_and_actions_rec_(node->right.get(), cur, paths_out, actions_out);
        cur.pop_back();
    }
};
