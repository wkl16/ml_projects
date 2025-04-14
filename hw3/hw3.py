# needs to be a executable python file set up so that TA can run it and corroborate results in the report.

def run_script(script):
    with open(script, 'r') as file: script_code = file.read()
    exec(script_code, globals())

if __name__ == "__main__":
    run_script('gen_data.py')
    run_script('svc_classification.py')
    run_script('eval_best_model.py')