#!/usr/bin/env python3
"""
Hardware detection script for Vintern-1B Demo
This script detects available hardware and sets the appropriate environment variables
"""

import platform
import os
import sys
import subprocess
import json

def detect_nvidia_gpu():
    """Detect NVIDIA GPU and return info if available"""
    try:
        # Try to run nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # NVIDIA GPU detected
            gpu_info = result.stdout.strip().split(',')
            return {
                'type': 'nvidia',
                'name': gpu_info[0].strip(),
                'memory': gpu_info[1].strip() if len(gpu_info) > 1 else 'Unknown',
                'driver': gpu_info[2].strip() if len(gpu_info) > 2 else 'Unknown'
            }
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    
    return None

def detect_amd_gpu():
    """Detect AMD GPU and return info if available"""
    # For Linux
    if platform.system() == 'Linux':
        try:
            # Check for AMD GPU using rocm-smi
            result = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and 'GPU' in result.stdout:
                # Extract GPU name
                for line in result.stdout.splitlines():
                    if 'GPU' in line and 'Product name' in line:
                        gpu_name = line.split(':')[-1].strip()
                        return {
                            'type': 'amd',
                            'name': gpu_name,
                            'memory': 'Unknown',
                            'driver': 'ROCm'
                        }
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
    
    # For Windows
    elif platform.system() == 'Windows':
        try:
            # Use PowerShell to get GPU info
            result = subprocess.run(
                ['powershell', '-Command', "Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                gpus = json.loads(result.stdout)
                if not isinstance(gpus, list):
                    gpus = [gpus]
                
                for gpu in gpus:
                    if 'AMD' in gpu.get('Name', ''):
                        return {
                            'type': 'amd',
                            'name': gpu.get('Name', 'AMD GPU'),
                            'memory': f"{int(gpu.get('AdapterRAM', 0)) / (1024**3):.2f} GB" if 'AdapterRAM' in gpu else 'Unknown',
                            'driver': 'Unknown'
                        }
        except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
            pass
    
    return None

def detect_apple_silicon():
    """Detect Apple Silicon and return info if available"""
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # This is Apple Silicon
        try:
            # Get more detailed info using sysctl
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                check=False
            )
            
            cpu_info = result.stdout.strip() if result.returncode == 0 else 'Apple Silicon'
            
            # Try to get memory info
            mem_result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if mem_result.returncode == 0 and mem_result.stdout.strip():
                try:
                    mem_bytes = int(mem_result.stdout.strip())
                    memory = f"{mem_bytes / (1024**3):.2f} GB"
                except ValueError:
                    memory = 'Unknown'
            else:
                memory = 'Unknown'
            
            return {
                'type': 'apple_silicon',
                'name': cpu_info,
                'memory': memory,
                'mps_available': True
            }
        except (FileNotFoundError, subprocess.SubprocessError):
            # Simplified detection
            return {
                'type': 'apple_silicon',
                'name': 'Apple Silicon',
                'memory': 'Unknown',
                'mps_available': True
            }
    
    return None

def detect_hardware():
    """Detect available hardware and return the best option"""
    system = platform.system()
    machine = platform.machine()
    
    # Detect all available hardware
    hardware = {
        'system': system,
        'architecture': machine,
        'apple_silicon': detect_apple_silicon(),
        'nvidia_gpu': detect_nvidia_gpu(),
        'amd_gpu': detect_amd_gpu()
    }
    
    # Determine the best hardware to use based on priorities
    if system == 'Darwin' and machine == 'arm64':
        # Mac with Apple Silicon
        if hardware['apple_silicon']:
            hardware['best'] = 'apple_silicon'
        elif hardware['nvidia_gpu']:
            hardware['best'] = 'nvidia'
        elif hardware['amd_gpu']:
            hardware['best'] = 'amd'
        else:
            hardware['best'] = 'cpu'
    else:
        # Windows/Linux or Mac with Intel/AMD
        if hardware['nvidia_gpu']:
            hardware['best'] = 'nvidia'
        elif hardware['amd_gpu']:
            hardware['best'] = 'amd'
        else:
            hardware['best'] = 'cpu'
    
    return hardware

def print_hardware_info(hardware):
    """Print hardware information in a readable format"""
    print(f"System: {hardware['system']} ({hardware['architecture']})")
    print(f"Best hardware detected: {hardware['best'].upper()}")
    print("")
    
    if hardware['apple_silicon']:
        print("Apple Silicon:")
        print(f"  Name: {hardware['apple_silicon']['name']}")
        print(f"  Memory: {hardware['apple_silicon']['memory']}")
        print(f"  MPS Available: {hardware['apple_silicon']['mps_available']}")
        print("")
    
    if hardware['nvidia_gpu']:
        print("NVIDIA GPU:")
        print(f"  Name: {hardware['nvidia_gpu']['name']}")
        print(f"  Memory: {hardware['nvidia_gpu']['memory']}")
        print(f"  Driver: {hardware['nvidia_gpu']['driver']}")
        print("")
    
    if hardware['amd_gpu']:
        print("AMD GPU:")
        print(f"  Name: {hardware['amd_gpu']['name']}")
        print(f"  Memory: {hardware['amd_gpu']['memory']}")
        print(f"  Driver: {hardware['amd_gpu']['driver']}")
        print("")
    
    if hardware['best'] == 'cpu':
        print("No GPU detected, using CPU.")
        print("")

def set_environment_variables(hardware):
    """Set environment variables based on detected hardware"""
    if hardware['best'] == 'apple_silicon':
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("Set PYTORCH_ENABLE_MPS_FALLBACK=1 for Apple Silicon")
    
    # Return the best hardware type for use in other scripts
    return hardware['best']

if __name__ == "__main__":
    hardware = detect_hardware()
    best_hardware = set_environment_variables(hardware)
    
    # If called with --json flag, print hardware info as JSON and exit
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        print(json.dumps(hardware))
        sys.exit(0)
    
    # Otherwise print human-readable format
    print_hardware_info(hardware)
    
    # Exit with code that indicates the best hardware
    # This can be used by shell scripts
    if best_hardware == 'apple_silicon':
        sys.exit(10)
    elif best_hardware == 'nvidia':
        sys.exit(20)
    elif best_hardware == 'amd':
        sys.exit(30)
    else:  # CPU
        sys.exit(40)
