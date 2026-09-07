import subprocess
import fire
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from UnlearnCanvas_resources.const import class_available
# from UnlearnCanvas_resources.const import class_available_subsample as class_available

def run_scripts_sequentially(
    classes_to_unlearn, input_dir, output_dir, style_ckpt, class_ckpt, batch_size, seed
):
    accuracy_script = os.path.join(SCRIPT_DIR, "accuracy_unlearncanvas_cls_fast.py")
    
    # Ensure the script exists
    if not os.path.exists(accuracy_script):
        print(f"Error: Script not found at {accuracy_script}")
        print(f"Current directory: {os.getcwd()}")
        return False
    
    base_command = (
        f"PYTHONPATH=. python {accuracy_script} "
        f"--input_dir '{input_dir}' "
        f"--output_dir '{output_dir}' "
        f"--style_ckpt '{style_ckpt}' "
        f"--class_ckpt '{class_ckpt}' "
        "--cls '{}' "
        f"--batch_size {batch_size} "
        f"--seed [{seed}]"
    )
    
    for cls in classes_to_unlearn:
        command = base_command.format(cls)
        print(f"Running command: {command}")
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            print(
                f"Error: Script failed with return code {process.returncode} for cls '{cls}'"
            )
            return False
        else:
            print(f"Successfully completed script for cls '{cls}'")
    
    return True

def main(input_dir, output_dir, style_ckpt, class_ckpt, batch_size, seed=188):
    success = run_scripts_sequentially(
        class_available, input_dir, output_dir, style_ckpt, class_ckpt, batch_size, seed
    )
    
    if not success:
        print("Error: Failed to run all classification scripts")
        return
    
    # Fix the path to the average accuracy script
    avg_script_path = os.path.join(SCRIPT_DIR, "avg_accuracy_cls.py")
    
    # Ensure the script exists
    if not os.path.exists(avg_script_path):
        print(f"Error: Average accuracy script not found at {avg_script_path}")
        print(f"Current directory: {os.getcwd()}")
        return
    
    process = subprocess.run(
        f"PYTHONPATH=. python {avg_script_path} '{output_dir}'",
        shell=True,
    )
    
    if process.returncode != 0:
        print("Error: Failed to run average accuracy calculation")

if __name__ == "__main__":
    fire.Fire(main)