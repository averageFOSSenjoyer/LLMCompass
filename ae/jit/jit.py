import json, re
from hardware_model.compute_module import (
    VectorUnit,
    SystolicArray,
    Core,
    ComputeModule,
    overhead_dict,
)
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule
from hardware_model.device import Device
from hardware_model.interconnect import LinkModule, InterConnectModule, TopologyType
from hardware_model.system import System
from software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from software_model.utils import data_type_dict, Tensor
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
from math import ceil

from design_space_exploration.dse import template_to_system, read_architecture_template
from multiprocessing import Process, Lock
import time
from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2

def jit(num_gpus, lock):
    input_seq_length = 1000
    batch_size = 8
    output_seq_length = 1000
    arch_specs = read_architecture_template("configs/24.json")
    arch_specs["device_count"] = num_gpus
    device_count = arch_specs["device_count"]

    model_init = TransformerBlockInitComputationTP(
        d_model=9216,
        n_heads=72,
        device_count=device_count,
        data_type=data_type_dict["fp16"],
    )
    model_auto_regression = TransformerBlockAutoRegressionTP(
        d_model=9216,
        n_heads=72,
        device_count=device_count,
        data_type=data_type_dict["fp16"],
    )
    _ = model_init(
        Tensor([batch_size, input_seq_length, model_init.d_model], data_type_dict["fp16"])
    )
    _ = model_auto_regression(
        Tensor([batch_size, 1, model_init.d_model], data_type_dict["fp16"]),
        input_seq_length + output_seq_length,
    )
    compute_area_mm2 = calc_compute_chiplet_area_mm2(arch_specs)
    io_area_mm2 = calc_io_die_area_mm2(arch_specs)
    arch_specs["device"]["io"]["memory_channel_physical_count"] = 24 // num_gpus * 6
    arch_specs["device"]["io"]["memory_channel_active_count"] = 24 // num_gpus * 6
    print(
        f"{compute_area_mm2}, {io_area_mm2}, {compute_area_mm2 + io_area_mm2}"
    )
    system = template_to_system(arch_specs)
    auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
        system, "heuristic-GPU"
    )
    print(
        f"{auto_regression_latency_simulated}"
    )
    init_latency_simulated = model_init.compile_and_simulate(system, "heuristic-GPU")
    print(
        f"{init_latency_simulated}"
    )
    with lock:
        with open(f"ae/jit/jit_results_bs{batch_size}_it{input_seq_length}_ot{output_seq_length}_pf.csv", "a") as f:
            f.write(
                f"{num_gpus}, {compute_area_mm2 + io_area_mm2}, {init_latency_simulated}, {model_init.simluate_log}\n"
            )
        with open(f"ae/jit/jit_results_bs{batch_size}_it{input_seq_length}_ot{output_seq_length}_ar.csv", "a") as f:
            f.write(
                f"{num_gpus}, {compute_area_mm2 + io_area_mm2}, {auto_regression_latency_simulated}, {model_auto_regression.simluate_log}\n"
            )


lock = Lock()
processes = [
    Process(target=jit, args=(num_gpus, lock,))
    for num_gpus in [1, 2, 3, 4, 6, 12, 24]
]

try:
    for p in processes:
        p.start()

    while any(p.is_alive() for p in processes):
        time.sleep(1)
except KeyboardInterrupt:
    print("Terminating processes...")
    for p in processes:
        p.terminate()
        p.join()


print("All processes have finished.")
