//! CUDA Device Detection Example
//!
//! This example demonstrates how to detect and query CUDA devices
//! in the TensorLogic backend. It shows:
//!
//! 1. Checking for CUDA availability
//! 2. Detecting available CUDA devices
//! 3. Querying device information (name, memory, compute capability)
//! 4. Using DeviceManager to access devices
//!
//! Note: This example will work even without CUDA installed - it will
//! gracefully report no CUDA devices found.

use tensorlogic_scirs_backend::{
    cuda_device_count, detect_cuda_devices, is_cuda_available, Device, DeviceManager, DeviceType,
};

fn main() {
    println!("=== TensorLogic CUDA Device Detection ===\n");

    // Check if CUDA is available on the system
    println!("1. Checking CUDA Availability");
    println!("   ----------------------------");
    if is_cuda_available() {
        println!("   ✓ CUDA is available on this system");
        println!("   Detected via: nvidia-smi or CUDA environment variables\n");
    } else {
        println!("   ✗ CUDA is not available on this system");
        println!("   This is normal if you don't have an NVIDIA GPU or CUDA installed\n");
    }

    // Detect CUDA devices
    println!("2. Detecting CUDA Devices");
    println!("   ----------------------");
    let cuda_devices = detect_cuda_devices();

    if cuda_devices.is_empty() {
        println!("   No CUDA devices found");
        println!("   The backend will use CPU for computation\n");
    } else {
        println!("   Found {} CUDA device(s):\n", cuda_devices.len());

        for device in &cuda_devices {
            println!("   GPU {}:", device.index);
            println!("     Name: {}", device.name);
            println!(
                "     Memory: {} MB ({:.2} GB)",
                device.memory_mb,
                device.memory_mb as f64 / 1024.0
            );

            if let Some((major, minor)) = device.compute_capability {
                println!("     Compute Capability: {}.{}", major, minor);

                // Provide context for compute capability
                let description = match major {
                    9 => "Latest generation (Hopper)",
                    8 => match minor {
                        9 => "High-end Ampere/Ada",
                        6 => "High-end Ampere",
                        0 => "Ampere architecture",
                        _ => "Ampere/Ada architecture",
                    },
                    7 => "Volta/Turing architecture",
                    6 => "Pascal architecture",
                    5 => "Maxwell architecture",
                    3 => "Kepler architecture",
                    _ => "Older architecture",
                };
                println!("     Architecture: {}", description);
            }
            println!();
        }
    }

    // Use DeviceManager to get all available devices
    println!("3. Device Manager");
    println!("   --------------");
    let manager = DeviceManager::new();

    println!(
        "   Available devices: {}",
        manager.available_devices().len()
    );
    for device in manager.available_devices() {
        println!("     - {}", device);
    }

    println!("\n   Default device: {}", manager.default_device());
    println!(
        "   GPU support: {}",
        if manager.has_gpu() { "Yes" } else { "No" }
    );

    // Count devices by type
    println!("\n   Device counts by type:");
    println!("     CPU:  {}", manager.count_devices(DeviceType::Cpu));
    println!("     CUDA: {}", manager.count_devices(DeviceType::Cuda));

    // Additional CUDA information
    println!("\n4. Quick CUDA Status");
    println!("   -----------------");
    let count = cuda_device_count();
    println!("   Total CUDA devices: {}", count);

    if count > 0 {
        println!("\n   Example: Creating a specific CUDA device:");
        let cuda_0 = Device::cuda(0);
        println!("     Device: {}", cuda_0);
        println!("     Type: {}", cuda_0.device_type());
        println!("     Index: {}", cuda_0.index());
        println!("     Is GPU: {}", cuda_0.is_gpu());

        println!("\n   Note: While CUDA devices are detected, full GPU execution");
        println!("   support requires scirs2 GPU features (planned for future release).");
    }

    println!("\n=== End of CUDA Detection Example ===");
}
