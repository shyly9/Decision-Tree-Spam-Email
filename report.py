from flask import Flask, render_template
from sklearn.metrics import classification_report
from C45 import DecisionTreeC45, X_test, y_test, X_train, y_train
import numpy as np
import joblib

report = Flask(__name__)

@report.route('/')
def index():
    loaded_model = joblib.load('decision_tree_model.joblib')

    # Dự đoán kết quả trên tập kiểm tra
    y_pred_c45 = loaded_model.predict(X_test)

    # Tạo thông tin cơ bản về mô hình
    basic_info = {
        'num_features': X_train.shape[1],
        'num_classes': len(np.unique(y_train)),
        'num_spam': np.sum(y_train == 1),
        'num_spam_test': np.sum(y_test == 1),
        'num_non_spam': np.sum(y_train == 0),
        'num_non_spam_test': np.sum(y_test == 0),
        'max_depth': loaded_model.max_depth,
        'min_samples_split': loaded_model.min_samples_split
    }

    # Tạo báo cáo phân loại
    classification_rep = classification_report(y_test, y_pred_c45, output_dict=True)

    # Tạo và hiển thị biểu đồ cây quyết định
    dot = loaded_model.display_tree()
    dot.render("static/C45_tree", format="png", cleanup=True)

    return render_template('report.html', basic_info=basic_info, classification_rep=classification_rep)

if __name__ == '__main__':
    report.run(debug=False, threaded=True, port=5001)
