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


input_seq_length = 100000
batch_size = 1024
output_seq_length = 2000
arch_specs = read_architecture_template("configs/24.json")
device_count = arch_specs["device_count"]
model_init = TransformerBlockInitComputationTP(
    d_model=65536,
    n_heads=120,
    device_count=device_count,
    data_type=data_type_dict["fp16"],
)
model_auto_regression = TransformerBlockAutoRegressionTP(
    d_model=65536,
    n_heads=120,
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

def test_topo(topo, bw, lock):
    arch_specs["interconnect"]["topology"] = topo
    arch_specs["interconnect"]["link"]["bandwidth_per_direction_byte"] = bw / 2
    arch_specs["interconnect"]["link"]["bandwidth_both_directions_byte"] = bw
    compute_area_mm2 = calc_compute_chiplet_area_mm2(arch_specs)
    io_area_mm2 = calc_io_die_area_mm2(arch_specs)
    print(
        f"{topo}, {bw}, {compute_area_mm2}, {io_area_mm2}, {compute_area_mm2 + io_area_mm2}"
    )
    system = template_to_system(arch_specs)
    auto_regression_latency_simulated = model_auto_regression.roofline_model(
        system, #"heuristic-GPU"
    )
    print(
        f"{topo}, {bw}, {auto_regression_latency_simulated}"
    )
    init_latency_simulated = model_init.roofline_model(
        system, #"heuristic-GPU"
    )
    print(
        f"{topo}, {bw}, {init_latency_simulated}"
    )
    with lock:
        with open(f"ae/topo/topo_results_bs{batch_size}_init.csv", "a") as f:
            f.write(
                f"{topo}, {bw}, {compute_area_mm2 + io_area_mm2}, {init_latency_simulated}, {model_init.simluate_log}\n"
            )
        with open(f"ae/topo/topo_results_bs{batch_size}_ar.csv", "a") as f:
            f.write(
                f"{topo}, {bw}, {compute_area_mm2 + io_area_mm2}, {auto_regression_latency_simulated}, {model_auto_regression.simluate_log}\n"
            )


lock = Lock()
processes = [
    Process(target=test_topo, args=(topo, bw, lock))
    for topo in ["FC", "RING"]
    for bw in [7e9, 35e9, 63e9, 91e9, 119e9]
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
