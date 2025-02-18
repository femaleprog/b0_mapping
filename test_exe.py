import subprocess

# Path to the compiled MATLAB executable
matlab_exe_path = "/volatile/home/st281428/nmrwizard/img_processing/BET"


# Run the executable
result = subprocess.run([matlab_exe_path], capture_output=True, text=True)

# Print the output from the MATLAB executable
print(result.stdout)
print(result.stderr)
