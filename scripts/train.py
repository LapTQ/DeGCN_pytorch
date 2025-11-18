NAMEF_CFG = "v206--satudora_veo3_awlrecord--r2.4-0xauto-1x1--satudora-filter-roi-conf--only-normal-satudora--veo3-all--1s-15frames--split-17-class--12-kpts"
num_models = 10
ls_devices = [0, 0, 2, 3, 0, 0, 2, 3, 0, 0]

import subprocess
import multiprocessing as mp
import concurrent.futures

# import time
# time.sleep(4 * 3600)  # sleep for 4 hours to wait for GPU availability

with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(
            subprocess.run,
            args=f"python main.py --config config/fs26/{NAMEF_CFG}.yaml --device {device} --overwrite --id_model {id_model}",
            # cwd="/home/laptq/laptq-fs26-shoplifting-detection/submodules/ProtoGCN",
            shell=True,
            check=True,
            text=True,
        )
        for id_model, device in zip(range(num_models), ls_devices)
    ]
    ls_trained_model = [f.result() for f in futures]