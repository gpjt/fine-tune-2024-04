import subprocess

with open("./results.csv", "w") as f:
    pass

for sequence_length in range(1, 2049):
    subprocess.check_call([
        "deepspeed",
        "measure_memory_usage_for_sequence_length.py",
        "--",
        str(sequence_length)
    ])
