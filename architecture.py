import shutil
import subprocess
import json
import os
import sys
import csv
import io

print("HARDWARE ARCHITECTURE CHECKER AND ANALYZER\n")

def ask_training_time():
    while True:
        training_time=input("Please input the time u have for training the model (in hours)--").strip()
        try:
            time=int(training_time)
            while True:
                if time <=0:
                    print("Training time must be greater than 0\n")
                    training_time=input("Please input the time u have for training the model (in hours)--").strip()
                    time=int(training_time)
                else:
                    break
            print(f"Your chosen time for training is: {time} hours.")
            while True:
                surity_check=input(f"Are you sure u want to allocate {time} hours for model training? (y/n) ").lower().strip()
                if surity_check in ("y","yes"):
                    return time
                elif surity_check in ("n","no"):
                    break
                else:
                    print("Invalid input.Please enter 'y' or 'n'\n")
        except ValueError:
            print("Please input a valid time limit.")


def scan_hardware():
    targeted_gpus=[]
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
                targeted_gpus.append(("cuda", vram))
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
                targeted_gpus.append(("modern-rocm", vram_mib))
        except(subprocess.SubprocessError,KeyError,json.JSONDecodeError):
            pass   
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        try:
            command_amd_legacy=["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"]
            output_cmd_amd_legacy=subprocess.run(command_amd_legacy, **run_settings)
            csv_data=output_cmd_amd_legacy.stdout.strip()
            reader=csv.DictReader(io.StringIO(csv_data))
            for row in reader:
                gpu_amd_legacy=row.get("Device Name") or row.get("card0") or "Unknown AMD GPU"
                vram_raw=row.get("VRAM Total Memory (B)")
                if vram_raw and vram_raw.isdigit():
                    vram_amd_legacy=int(vram_raw)
                    if vram_amd_legacy>1_000_000:
                        vram_legacy_mib=vram_amd_legacy//(1024*1024)
                    else:
                        vram_legacy_mib=vram_amd_legacy
                    print(f"GPU is {gpu_amd_legacy}.The VRAM is {vram_legacy_mib}")
                    targeted_gpus.append(("legacy-rocm", vram_legacy_mib))
            print("ROCm ARCHITECTURE DETECTED.")
        except (subprocess.SubprocessError,json.JSONDecodeError):
            pass    
    if len(targeted_gpus)==0:
        return ("cpu", 0)
    if len(targeted_gpus)==1:
        return targeted_gpus[0]
    if len(targeted_gpus)>=2:
        for index,(architecture,vram) in enumerate(targeted_gpus):
            print(f"[{index}] Architecture:{architecture}, VRAM:{vram}MB")
        while True:
            try:
                gpu_choice=int(input("SELECT A GPU INDEX NUMBER FOR TRAINING:\n"))
                if 0 <= gpu_choice < len(targeted_gpus):
                    return targeted_gpus[gpu_choice]
                print("INVALID INDEX NUMBER. PLEASE SELECT A VALID INDEX.")
            except ValueError:
                print("PLEASE ENTER A VALID NUMBER.")
detected_hardware=scan_hardware()
allocated_time=ask_training_time()

def identify_os():
    base_path=os.path.join("sizer_core", "target", "release", "sizer_core")
    if sys.platform=="win32":
        return base_path+".exe"
    return base_path
    
if detected_hardware[0]=="cpu":
    print("NO DEDICATED AI-GPU DRIVERS DETECTED.")
else:
    sizer_binary=identify_os()
    if sys.platform !="win32" and os.path.exists(sizer_binary):
        os.chmod (sizer_binary, 0o755)
    while True:
        try:
            result=subprocess.run([sizer_binary]
            ,input=f"{allocated_time}\n,{detected_hardware[1]}\n",text=True,check=True,capture_output=True)
            total_params=result.stdout.strip()
            break
        except(subprocess.CalledProcessError,FileNotFoundError):
            print("\nENSURE THAT U HAVE FOLLOWED ALL INSTRUCTIONS AND HAVE RUST INSTALLED AND (sizer_core) SET UP.")
            input("AFTER COMPLETING PRESS ENTER:")