#!/usr/bin/env python3

import time
import json
import requests
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Optional

class GPUMonitor:
    def __init__(self, service_url: str = "http://localhost:5003"):
        self.service_url = service_url
        self.monitoring = False
        
    def get_nvidia_smi_data(self) -> Optional[Dict]:
        """Get GPU data from nvidia-smi"""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            gpu_data = []
            
            for line in lines:
                if line.strip():
                    fields = [field.strip() for field in line.split(',')]
                    gpu_info = {
                        'timestamp': fields[0],
                        'name': fields[1],
                        'pci_bus_id': fields[2],
                        'driver_version': fields[3],
                        'pstate': fields[4],
                        'pcie_link_gen_max': fields[5],
                        'pcie_link_gen_current': fields[6],
                        'temperature': int(fields[7]) if fields[7] != '[Not Supported]' else None,
                        'gpu_utilization': int(fields[8]) if fields[8] != '[Not Supported]' else 0,
                        'memory_utilization': int(fields[9]) if fields[9] != '[Not Supported]' else 0,
                        'memory_total': int(fields[10]) if fields[10] != '[Not Supported]' else 0,
                        'memory_free': int(fields[11]) if fields[11] != '[Not Supported]' else 0,
                        'memory_used': int(fields[12]) if fields[12] != '[Not Supported]' else 0,
                    }
                    gpu_data.append(gpu_info)
            
            return gpu_data[0] if gpu_data else None
            
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError) as e:
            print(f"Error getting GPU data: {e}")
            return None
    
    def get_service_status(self) -> Optional[Dict]:
        """Get service status from the Flask app"""
        try:
            response = requests.get(f"{self.service_url}/gpu-status", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting service status: {e}")
            return None
    
    def get_service_health(self) -> Optional[Dict]:
        """Get service health from the Flask app"""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting service health: {e}")
            return None
    
    def format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
    
    def print_status(self, gpu_data: Dict, service_data: Dict, health_data: Dict):
        """Print current status"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n{'='*60}")
        print(f"GPU Monitor - {timestamp}")
        print(f"{'='*60}")
        
        # Service Health
        print(f"Service Status: {'🟢 Online' if health_data and health_data.get('status') == 'ok' else '🔴 Offline'}")
        
        if service_data:
            print(f"GPU Available: {'✅ Yes' if service_data.get('gpu_available') else '❌ No'}")
            print(f"Device: {service_data.get('device', 'Unknown')}")
            print(f"Whisper Model: {service_data.get('whisper_model_size', 'Unknown')} ({'Loaded' if service_data.get('whisper_model_loaded') else 'Not Loaded'})")
        
        # GPU Information
        if gpu_data:
            print(f"\nGPU: {gpu_data['name']}")
            print(f"Driver: {gpu_data['driver_version']}")
            print(f"Temperature: {gpu_data['temperature']}°C" if gpu_data['temperature'] else "Temperature: N/A")
            print(f"Power State: {gpu_data['pstate']}")
            
            # Utilization
            gpu_util = gpu_data['gpu_utilization']
            mem_util = gpu_data['memory_utilization']
            print(f"GPU Utilization: {gpu_util}% {'🔥' if gpu_util > 80 else '🟡' if gpu_util > 50 else '🟢'}")
            print(f"Memory Utilization: {mem_util}% {'🔥' if mem_util > 80 else '🟡' if mem_util > 50 else '🟢'}")
            
            # Memory
            mem_total = gpu_data['memory_total'] * 1024 * 1024  # Convert MB to bytes
            mem_used = gpu_data['memory_used'] * 1024 * 1024
            mem_free = gpu_data['memory_free'] * 1024 * 1024
            
            print(f"Memory Used: {self.format_bytes(mem_used)} / {self.format_bytes(mem_total)}")
            print(f"Memory Free: {self.format_bytes(mem_free)}")
            
            # Service-specific memory info
            if service_data and 'gpu_memory_allocated' in service_data:
                torch_allocated = service_data['gpu_memory_allocated']
                torch_cached = service_data['gpu_memory_cached']
                print(f"PyTorch Allocated: {self.format_bytes(torch_allocated)}")
                print(f"PyTorch Cached: {self.format_bytes(torch_cached)}")
        else:
            print("\n❌ GPU data unavailable")
    
    def monitor_continuous(self, interval: int = 5):
        """Continuously monitor GPU and service"""
        print("Starting continuous GPU monitoring... (Press Ctrl+C to stop)")
        self.monitoring = True
        
        try:
            while self.monitoring:
                gpu_data = self.get_nvidia_smi_data()
                service_data = self.get_service_status()
                health_data = self.get_service_health()
                
                # Clear screen (works on most terminals)
                print("\033[H\033[J", end="")
                
                self.print_status(gpu_data, service_data, health_data)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            self.monitoring = False
    
    def monitor_once(self):
        """Get current status once"""
        gpu_data = self.get_nvidia_smi_data()
        service_data = self.get_service_status()
        health_data = self.get_service_health()
        
        self.print_status(gpu_data, service_data, health_data)
    
    def log_to_file(self, filename: str, interval: int = 30, duration: int = 3600):
        """Log monitoring data to file"""
        print(f"Logging to {filename} for {duration} seconds...")
        
        start_time = time.time()
        
        with open(filename, 'w') as f:
            f.write("timestamp,gpu_utilization,memory_utilization,memory_used_mb,temperature,service_status\n")
            
            while time.time() - start_time < duration:
                gpu_data = self.get_nvidia_smi_data()
                health_data = self.get_service_health()
                
                timestamp = datetime.now().isoformat()
                
                if gpu_data:
                    gpu_util = gpu_data['gpu_utilization']
                    mem_util = gpu_data['memory_utilization']
                    mem_used = gpu_data['memory_used']
                    temp = gpu_data['temperature'] or 0
                else:
                    gpu_util = mem_util = mem_used = temp = 0
                
                service_status = 1 if health_data and health_data.get('status') == 'ok' else 0
                
                f.write(f"{timestamp},{gpu_util},{mem_util},{mem_used},{temp},{service_status}\n")
                f.flush()
                
                time.sleep(interval)
        
        print(f"Logging completed. Data saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="GPU Performance Monitor for Video Caption Service")
    parser.add_argument("--url", default="http://localhost:5003", help="Service URL")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--log", help="Log to file for specified duration")
    parser.add_argument("--duration", type=int, default=3600, help="Logging duration in seconds")
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(args.url)
    
    if args.once:
        monitor.monitor_once()
    elif args.log:
        monitor.log_to_file(args.log, args.interval, args.duration)
    else:
        monitor.monitor_continuous(args.interval)

if __name__ == "__main__":
    main()