import subprocess

with open("./results.csv", "w") as f:
    pass

for sequence_length in range(1, 2049, 10):
    succeeded = False
    tries = 0
    while not succeeded and tries < 5:
        tries += 1
        try:
            subprocess.check_call([
                "deepspeed",
                "measure_memory_usage_for_sequence_length.py",
                "--",
                str(sequence_length)
            ])
            succeeded = True
        except subprocess.CalledProcessError as exc:
            print(f"************************** ERROR {exc}")

    if not succeeded:
        print("***************** Too many failures, crapping out")
        break
