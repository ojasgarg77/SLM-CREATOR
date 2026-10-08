import shutil
import subprocess
import json
import os
import sys

print("HARDWARE ARCHITECTURE CHECKER AND ANALYZER\n")
def scan_hardware():
    run_settings={
        "check":True,
        "stdout":subprocess.PIPE,
        "stderr":subprocess.DEVNULL,
        "text":True,
        "encoding":"utf-8"
    }

    nvidia_bin=shutil.which("nvidia-smi")
    if not nvidia_bin:
        fallback_path=("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe")
        if os.path.exists(fallback_path):   
            nvidia_bin=fallback_path
    if nvidia_bin:
        try:
            command_nvidia=[ nvidia_bin, "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"]
            output_cmd_nvidia=subprocess.run(command_nvidia, **run_settings)
            output_nvidia=output_cmd_nvidia.stdout.strip().splitlines()
            for line in output_nvidia:
                gpu_conditions=line.strip().split(',')
                gpu=gpu_conditions[0]
                vram=int(gpu_conditions[1])
                print(f"GPU is {gpu}.The VRAM is {vram}")
            print("CUDA ARCHITECTURE DETECTED.")
            return "cuda"
        except subprocess.SubprocessError:
            pass
    if shutil.which("amd-smi"):
        try:
            command_amd=["amd-smi", "static", "--json"]
            output_cmd_amd=subprocess.run(command_amd, **run_settings)
            output_amd=json.loads(output_cmd_amd.stdout)
            devices=output_amd.get("devices",[])
            for device in devices:
                gpu_amd=device["asic"]["market_name"]
                vram_amd=int(device["vram"]["total"])
                if int(vram_amd)>1_000_000:
                    vram_mib=vram_amd//(1024*1024)
                else:
                    vram_mib=vram_amd
                print(f"GPU is {gpu_amd}.The VRAM is {vram_mib}")
            return "modern-rocm"
        except(subprocess.SubprocessError,KeyError,json.JSONDecodeError):
            pass   
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        try:
            command_amd_legacy=["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"]
            output_cmd_amd_legacy=subprocess.run(command_amd_legacy, **run_settings)
            output_amd_legacy=output_cmd_amd_legacy.stdout.splitlines()
            for line in output_amd_legacy[1:]:
                if line.strip():
                    parts=line.split(",")
                    gpu_amd_legacy=parts[3]
                    vram_amd_legacy=int(parts[1])
                    if vram_amd_legacy>1_000_000:
                        vram_legacy_mib=vram_amd_legacy//(1024*1024)
                    else:
                        vram_legacy_mib=vram_amd_legacy
                    print(f"GPU is {gpu_amd_legacy}.The VRAM is {vram_legacy_mib}")
            print("ROCm ARCHITECTURE DETECTED.")
            return "legacy-rocm"
        except (subprocess.SubprocessError,json.JSONDecodeError):
            pass    
    return "cpu"
    
detected_hardware=scan_hardware()

def identify_os():
    base_path=os.path.join("sizer_core", "target", "release", "sizer_core")
    if sys.platform=="win32":
        return base_path+".exe"
    return base_path
    
if detected_hardware=="cpu":
    print("NO DEDICATED AI-GPU DRIVERS DETECTED.")
else:
    sizer_binary=identify_os()
    if sys.platform !="win32" and os.path.exists(sizer_binary):
        os.chmod (sizer_binary, 0o755)
    while True:
        try:
            subprocess.run([sizer_binary],check=True)
            with open("model_config.json","r") as f:
                config=json.load(f)
            print(f"Target Parameters:{config['total_params']}")
            break
        except(subprocess.CalledProcessError,FileNotFoundError):
            print("\nENSURE THAT U HAVE FOLLOWED ALL INSTRUCTIONS AND HAVE RUST INSTALLED.")
            input("AFTER COMPLETING PRESS ENTER:")