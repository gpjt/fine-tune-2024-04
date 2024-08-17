from itertools import count

import subprocess

with open("./results.csv", "w") as f:
    pass

for batch_size in count(1):
    succeeded = False
    tries = 0
    while not succeeded and tries < 5:
        tries += 1
        try:
            subprocess.check_call([
                "deepspeed",
                "measure_memory_usage_for_batch_size.py",
                "--",
                str(batch_size)
            ])
            succeeded = True
        except subprocess.CalledProcessError as exc:
            print(f"************************** ERROR {exc}")

    with open("./results.csv", "r") as f:
        lines = f.readlines()
        if "OOM" in lines[-1]:
            print("Hit an OOM, exiting.")
            break

    if not succeeded:
        print("***************** Too many failures, crapping out")
        break
