import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from generate_data import generate_data_numbers, generate_data_fashion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def evaluate_model_with_params(dataset='numbers', kernel='rbf', best_params=None, pca_components=100):
    if dataset == 'numbers':
        X_train, X_test, y_train, y_test = generate_data_numbers()
    else:
        X_train, X_test, y_train, y_test = generate_data_fashion()
    
    # rebuild model pipeline 
    svc_params = {key.replace('svc__', ''): val for key, val in best_params.items()}
    svc = SVC(kernel=kernel, random_state=1, max_iter=10000, **svc_params)

    pipeline = Pipeline([
        ('pca', PCA(n_components=pca_components)),
        ('scaler', StandardScaler()),
        ('svc', svc)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fashion-MNIST: {kernel} kernel, {pca_components} scale")
    plt.show()

if __name__ == "__main__":
    # set params 
    best_params = {
        'svc__C': 10,
        'svc__gamma': 0.01,
        # 'svc__degree': 3  
    }

    evaluate_model_with_params(dataset='fashion', kernel='rbf', best_params=best_params)
