import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import sys
import joblib
import graphviz

# Đọc dữ liệu từ tệp csv
url = "spambase.csv"
data = pd.read_csv(url, sep=',')

data.drop_duplicates(inplace = True)

columns = data.columns[:-1].tolist()

# Chia dữ liệu thành features (X) và target (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


# Hàm tính entropy
def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


# Lớp Node trong cây quyết định
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # Khởi tạo một nút trong cây quyết định
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


# Lớp cây quyết định C4.5
class DecisionTreeC45:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None, feature_names=None):
        # Khởi tạo cây quyết định C4.5 với các tham số mặc định
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.feature_names = feature_names
        self.root = None

    def fit(self, X, y):
        # Huấn luyện cây quyết định trên tập dữ liệu huấn luyện
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)
        print("Decision Tree C4.5 Training Completed")

    def predict(self, X):
        # Dự đoán nhãn cho tập dữ liệu đầu vào
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        # Xây dựng cây quyết định C4.5 sử dụng thuật toán đệ quy
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Tiêu chí dừng
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split: #độ sâu đạt độ sâu tối đa/chỉ có một nhãn/số lượng mẫu nhỏ hơn số lượng mẫu tối thiểu
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Chọn ngẫu nhiên các đặc trưng để xem xét khi phân chia
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # Chọn phân chia tốt nhất dựa trên tỷ lệ lợi thông tin
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # Xây dựng cây con từ các nhánh phân chia
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        # Tìm tiêu chí phân chia tốt nhất dựa trên tỷ lệ lợi thông tin
        best_gain_ratio = -1
        split_idx, split_thresh = None, None

        # Duyệt qua các đặc trưng được chọn ngẫu nhiên
        for feat_idx in feat_idxs:
            # Lấy cột dữ liệu của đặc trưng hiện tại
            X_column = X[:, feat_idx]

            # Lấy các ngưỡng giá trị duy nhất trong đặc trưng
            thresholds = np.unique(X_column)

            # Duyệt qua các ngưỡng giá trị
            for threshold in thresholds:
                # Tính tỷ lệ lợi thông tin cho phân chia
                gain_ratio = self._information_gain_ratio(y, X_column, threshold)

                # So sánh và cập nhật nếu tỷ lệ lợi thông tin là tốt nhất
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain_ratio(self, y, X_column, split_thresh):
        # Phương thức tính gain ratio
        # parent loss (lỗi ở mức cha)
        parent_entropy = entropy(y)

        # generate split(tạo phân chia)
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children/ tính trung bình có trọng số của lỗi cho các nút con
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain
        ig = parent_entropy - child_entropy

        # split information
        split_info = -((n_l / n) * np.log2(n_l / n) + (n_r / n) * np.log2(n_r / n))

        # gain ratio = information gain / split information
        gain_ratio = ig / split_info

        return gain_ratio

    def _split(self, X_column, split_thresh):
        # Tạo danh sách chỉ mục của các mẫu thuộc về nút con bên trái và bên phải
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        # Duyệt cây để dự đoán nhãn cho một điểm dữ liệu mới
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        # Trả về nhãn xuất hiện nhiều nhất trong tập dữ liệu
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def display_tree(self, dot=None, node=None, parent_name="Root", edge_label=None):
        # Tạo biểu đồ của cây quyết định bằng cách sử dụng thư viện Graphviz.
        if node is None:
            node = self.root

        # Tạo biểu đồ mới nếu đang xử lý nút gốc
        if dot is None:
            dot = graphviz.Digraph(comment="Decision Tree")

        # Tạo nút và kết nối với nút cha (nếu có)
        dot.node(str(node), label=self._get_node_label(node))
        if edge_label is not None:
            dot.edge(parent_name, str(node), label=edge_label)

        # Nếu là nút lá, kết thúc
        if node.is_leaf_node():
            return dot

        # Đệ quy với cây con
        left_label = f"True"
        dot = self.display_tree(dot, node.left, str(node), left_label)
        right_label = f"False"
        dot = self.display_tree(dot, node.right, str(node), right_label)

        return dot

    def _get_node_label(self, node):
        # Trả về nhãn của một nút trong cây quyết định dùng để hiển thị trên biểu đồ.
        if node.is_leaf_node():
            if node.value == 1:
                return f"Class: spam"
            return f"Class: non-spam"
        else:
            feature_name = self.feature_names[node.feature]
            return f"{feature_name} <= {node.threshold}\nGain Ratio: {self._information_gain_ratio(y_train, X_train[:, node.feature], node.threshold):.4f}"

# Tạo mô hình và huấn luyện
clf_c45 = DecisionTreeC45(max_depth=10, feature_names=columns)
clf_c45.fit(X_train, y_train)

# Lưu mô hình vào file decision_tree_model.joblib
joblib.dump(clf_c45, 'decision_tree_model.joblib')