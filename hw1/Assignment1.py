def run_script(script):
    with open(script, 'r') as file: script_code = file.read()
    exec(script_code, globals())

if __name__ == "__main__":
    print("Executing Task 1")
    print("Plotting Perceptron…")
    run_script("perceptron.py")
    print("Plotting Adaline…")
    run_script("adaline.py")
    print("Executing Task 3")
    print("Running Multiclass Demo…")
    run_script("multiclassdemo.py")
    print("Executing Task 4")
    print("Plotting Adaline GD, SGD, and Mini-batch SGD")
    run_script("PlotAdaline.py")
    print("Assignment 1 finished")