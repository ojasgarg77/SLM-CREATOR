import shutil
import subprocess
import json

def scan_hardware():
    run_settings={
        "check":True,
        "stdout":subprocess.PIPE,
        "stderr":subprocess.DEVNULL,
        "text":True
    }
    if shutil.which("nvidia-smi"):
        try:
            command_nvidia=["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"]
            output_cmd_nvidia=subprocess.run(command_nvidia, **run_settings)
            output_nvidia=output_cmd_nvidia.stdout.strip().splitlines()
            for line in output_nvidia:
                gpu_conditions=line.strip().split(',')
                gpu=gpu_conditions[0]
                vram=gpu_conditions[1]
                print(f"GPU is {gpu}.The VRAM is{vram}")
            print("CUDA ARCHITECTURE DETECTED.")
            return "cuda"
        except subprocess.SubprocessError:
            pass
    elif shutil.which("rocm-smi") or shutil.which("rocminfo"):
        try:
            command_amd_legacy=["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"]
            output_cmd_amd_legacy=subprocess.run(command_amd_legacy, **run_settings)
            output_amd_legacy=output_cmd_amd_legacy.stdout.splitlines()
            for line in output_amd_legacy[1:]:
                if line.strip():
                    parts=line.split(",")
                    gpu_amd_legacy=parts[3]
                    vram_amd_legacy=parts[1]
                    vram_legacy_mib=int(vram_amd_legacy)//(1024*1024)
                    print(gpu_amd_legacy,vram_legacy_mib)
            print("ROCm ARCHITECTURE DETECTED.")
            return "legacy-rocm"
        except subprocess.SubprocessError:
            pass    
    elif shutil.which("amd-smi"):
        command_amd=["amd-smi", "static", "--json"]
        output_cmd_amd=subprocess.run(command_amd, **run_settings)
        output_amd=json.loads(output_cmd_amd.stdout)
        for device in output_amd["devices"]:
            gpu_amd=device["asic"]["market_name"]
            vram_amd=device["vram"]["total"]
            vram_mib=int(vram_amd)//(1024*1024)
            print(f"GPU is {gpu_amd}.The VRAM is {vram_mib}")
        return "modern-rocm"
    print("NO DEDICATED AI-GPU DRIVERS DETECTED.")
    return "cpu"
    
detected_hardware=scan_hardware()
