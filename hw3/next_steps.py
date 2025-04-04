from generate_data import generate_data_numbers
from generate_data import generate_data_fashion
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data_numbers()
    
    # Assuming this is the next step of the assignment here
    A3_pipe = make_pipeline(StandardScaler(), SVC(random_state=42))
    print(A3_pipe.fit(X_train, y_train).score(X_test, y_test))
    