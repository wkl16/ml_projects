def run_script(script):
    with open(script, 'r') as file: script_code = file.read()
    exec(script_code, globals())

if __name__ == "__main__":
    run_script("problem1/rl_eval.py")
    run_script("problem1/run_eval.py")
    run_script("qlearn_24.py")
