//! Hardware fingerprint collection for proof reproducibility.
//!
//! Captures memory type/speed, storage, NUMA topology, kernel version, and
//! microcode revision so proof results can be compared across instances with
//! full knowledge of the hardware environment.

#[cfg(target_os = "linux")]
use std::process::Command;

/// Complete hardware fingerprint for a proof run.
#[derive(Debug, Clone)]
pub struct HwFingerprint {
    /// CPU model string (from /proc/cpuinfo or lscpu)
    pub cpu_model: String,
    /// Logical core count
    pub cores: u32,
    /// Physical core count (may differ from logical with SMT)
    pub physical_cores: u32,
    /// CPU base frequency in MHz
    pub cpu_mhz: f64,
    /// Total RAM in bytes
    pub ram_bytes: u64,
    /// Memory type (e.g., "DDR4", "DDR5")
    pub mem_type: String,
    /// Memory speed in MT/s (e.g., 3200, 4800)
    pub mem_speed_mt: u32,
    /// Memory channels detected (or estimated)
    pub mem_channels: u32,
    /// True if mem_channels was estimated (dmidecode unreliable on VMs)
    pub mem_channels_estimated: bool,
    /// Theoretical peak memory bandwidth in GB/s
    pub mem_bandwidth_gbps: f64,
    /// NUMA node count
    pub numa_nodes: u32,
    /// L1d / L1i / L2 / L3 cache sizes
    pub cache_l1d: String,
    pub cache_l1i: String,
    pub cache_l2: String,
    pub cache_l3: String,
    /// ISA extensions (e.g., "avx2 avx512f neon sve")
    pub isa_extensions: String,
    /// Root block device type
    pub storage_type: String,
    /// Root volume size
    pub storage_size: String,
    /// Linux kernel version
    pub kernel_version: String,
    /// CPU microcode revision
    pub microcode: String,
}

impl HwFingerprint {
    /// Collect hardware fingerprint from the current system.
    pub fn collect() -> Self {
        let mut fp = HwFingerprint {
            cpu_model: String::new(),
            cores: 0,
            physical_cores: 0,
            cpu_mhz: 0.0,
            ram_bytes: 0,
            mem_type: "unknown".into(),
            mem_speed_mt: 0,
            mem_channels: 0,
            mem_channels_estimated: false,
            mem_bandwidth_gbps: 0.0,
            numa_nodes: 1,
            cache_l1d: String::new(),
            cache_l1i: String::new(),
            cache_l2: String::new(),
            cache_l3: String::new(),
            isa_extensions: String::new(),
            storage_type: String::new(),
            storage_size: String::new(),
            kernel_version: String::new(),
            microcode: String::new(),
        };

        #[cfg(target_os = "linux")]
        {
            fp.collect_cpuinfo();
            fp.collect_meminfo();
            fp.collect_lscpu();
            fp.collect_dmidecode();
            fp.collect_storage();
            fp.collect_kernel();
        }

        // Compute theoretical bandwidth if we have type + speed + channels.
        // Formula: speed_MT/s × 8 bytes/transfer × channels / 1000 = GB/s
        // (mem_speed_mt is already in MT/s, i.e., millions of transfers/s,
        //  so ×8 gives MB/s per channel, ÷1000 converts to GB/s)
        if fp.mem_speed_mt > 0 && fp.mem_channels > 0 {
            fp.mem_bandwidth_gbps =
                fp.mem_speed_mt as f64 * 8.0 * fp.mem_channels as f64 / 1000.0;
        }

        // Sanity check: on VMs, dmidecode often reports only 1 DIMM even when
        // the physical host has many memory channels. Any server with > 8 GB RAM
        // and only 1 detected channel is almost certainly a VM hiding real topology.
        if fp.mem_speed_mt > 0
            && fp.mem_channels <= 1
            && fp.ram_bytes > 8 * 1024 * 1024 * 1024
        {
            // Server platforms: 8 memory channels per socket is standard
            // (Cascade Lake, Ice Lake, Sapphire Rapids, Granite Rapids,
            //  EPYC Milan/Genoa/Turin, Graviton 2/3/4)
            let sockets = fp.numa_nodes.max(1);
            let estimated_channels = sockets * 8;
            fp.mem_channels = estimated_channels;
            fp.mem_channels_estimated = true;
            fp.mem_bandwidth_gbps =
                fp.mem_speed_mt as f64 * 8.0 * estimated_channels as f64 / 1000.0;
        }

        fp
    }

    /// Memory normalization factor relative to DDR4-3200 single channel.
    /// Returns a multiplier: DDR4-3200 1ch = 1.0, DDR5-4800 1ch = 1.5, etc.
    /// Used to normalize bandwidth-bound benchmarks.
    pub fn mem_bw_factor(&self) -> f64 {
        if self.mem_bandwidth_gbps > 0.0 {
            // Reference: DDR4-3200 × 1 channel = 25.6 GB/s
            self.mem_bandwidth_gbps / 25.6
        } else {
            1.0 // unknown → no normalization
        }
    }

    /// Compute normalization factor for FLOPS-bound workloads.
    /// Normalizes to "GFLOPS per core per GHz" to remove frequency/core-count effects.
    pub fn compute_factor(&self, measured_gflops: f64) -> f64 {
        let cores = if self.physical_cores > 0 { self.physical_cores } else { self.cores.max(1) };
        let ghz = if self.cpu_mhz > 0.0 { self.cpu_mhz / 1000.0 } else { 1.0 };
        measured_gflops / (cores as f64 * ghz)
    }

    /// Print the fingerprint to stdout.
    pub fn print(&self) {
        println!("Hardware Fingerprint:");
        println!("  CPU:          {}", self.cpu_model);
        println!("  Cores:        {} logical, {} physical", self.cores, self.physical_cores);
        if self.cpu_mhz > 0.0 {
            println!("  Frequency:    {:.0} MHz", self.cpu_mhz);
        }
        println!("  RAM:          {:.1} GB", self.ram_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
        let ch_note = if self.mem_channels_estimated { " (estimated)" } else { "" };
        println!("  Memory:       {} @ {} MT/s, {} channel(s){}", self.mem_type, self.mem_speed_mt, self.mem_channels, ch_note);
        if self.mem_bandwidth_gbps > 0.0 {
            println!("  Peak BW:      {:.1} GB/s (theoretical)", self.mem_bandwidth_gbps);
        }
        println!("  NUMA:         {} node(s)", self.numa_nodes);
        if !self.cache_l1d.is_empty() {
            println!("  Cache:        L1d={} L1i={} L2={} L3={}", self.cache_l1d, self.cache_l1i, self.cache_l2, self.cache_l3);
        }
        if !self.isa_extensions.is_empty() {
            println!("  ISA:          {}", self.isa_extensions);
        }
        if !self.storage_type.is_empty() {
            println!("  Storage:      {} ({})", self.storage_type, self.storage_size);
        }
        if !self.kernel_version.is_empty() {
            println!("  Kernel:       {}", self.kernel_version);
        }
        if !self.microcode.is_empty() {
            println!("  Microcode:    {}", self.microcode);
        }
    }

    /// Emit JSON fields (without surrounding braces).
    pub fn emit_json_fields(&self) {
        let e = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        println!("  \"hw_cpu_model\": \"{}\",", e(&self.cpu_model));
        println!("  \"hw_cores_logical\": {},", self.cores);
        println!("  \"hw_cores_physical\": {},", self.physical_cores);
        println!("  \"hw_cpu_mhz\": {:.0},", self.cpu_mhz);
        println!("  \"hw_ram_bytes\": {},", self.ram_bytes);
        println!("  \"hw_mem_type\": \"{}\",", e(&self.mem_type));
        println!("  \"hw_mem_speed_mt\": {},", self.mem_speed_mt);
        println!("  \"hw_mem_channels\": {},", self.mem_channels);
        println!("  \"hw_mem_channels_estimated\": {},", self.mem_channels_estimated);
        println!("  \"hw_mem_bandwidth_gbps\": {:.1},", self.mem_bandwidth_gbps);
        println!("  \"hw_numa_nodes\": {},", self.numa_nodes);
        println!("  \"hw_cache_l1d\": \"{}\",", e(&self.cache_l1d));
        println!("  \"hw_cache_l2\": \"{}\",", e(&self.cache_l2));
        println!("  \"hw_cache_l3\": \"{}\",", e(&self.cache_l3));
        println!("  \"hw_isa\": \"{}\",", e(&self.isa_extensions));
        println!("  \"hw_storage\": \"{}\",", e(&self.storage_type));
        println!("  \"hw_kernel\": \"{}\",", e(&self.kernel_version));
        println!("  \"hw_microcode\": \"{}\",", e(&self.microcode));
        println!("  \"hw_mem_bw_factor\": {:.3},", self.mem_bw_factor());
    }

    #[cfg(target_os = "linux")]
    fn collect_cpuinfo(&mut self) {
        let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") else { return };

        if let Some(line) = info.lines().find(|l| l.starts_with("model name")) {
            self.cpu_model = line.split(':').nth(1).unwrap_or("").trim().to_string();
        }
        // ARM doesn't have "model name" — try "Hardware" or lscpu later
        if self.cpu_model.is_empty() {
            if let Some(line) = info.lines().find(|l| l.starts_with("Hardware")) {
                self.cpu_model = line.split(':').nth(1).unwrap_or("").trim().to_string();
            }
        }

        self.cores = info.lines().filter(|l| l.starts_with("processor")).count() as u32;

        // x86 microcode
        if let Some(line) = info.lines().find(|l| l.starts_with("microcode")) {
            self.microcode = line.split(':').nth(1).unwrap_or("").trim().to_string();
        }

        // x86 frequency
        if let Some(line) = info.lines().find(|l| l.starts_with("cpu MHz")) {
            if let Some(val) = line.split(':').nth(1) {
                self.cpu_mhz = val.trim().parse().unwrap_or(0.0);
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn collect_meminfo(&mut self) {
        let Ok(info) = std::fs::read_to_string("/proc/meminfo") else { return };
        if let Some(line) = info.lines().find(|l| l.starts_with("MemTotal")) {
            // "MemTotal:       16384000 kB"
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                self.ram_bytes = parts[1].parse::<u64>().unwrap_or(0) * 1024;
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn collect_lscpu(&mut self) {
        let Ok(output) = Command::new("lscpu").output() else { return };
        let text = String::from_utf8_lossy(&output.stdout);

        for line in text.lines() {
            let parts: Vec<&str> = line.splitn(2, ':').collect();
            if parts.len() < 2 { continue; }
            let key = parts[0].trim();
            let val = parts[1].trim();

            match key {
                "Core(s) per socket" => {
                    let cores_per_socket: u32 = val.parse().unwrap_or(0);
                    // Multiply by socket count later
                    self.physical_cores = cores_per_socket;
                }
                "Socket(s)" => {
                    let sockets: u32 = val.parse().unwrap_or(1);
                    if self.physical_cores > 0 {
                        self.physical_cores *= sockets;
                    }
                }
                "NUMA node(s)" => {
                    self.numa_nodes = val.parse().unwrap_or(1);
                }
                "L1d cache" => self.cache_l1d = val.to_string(),
                "L1i cache" => self.cache_l1i = val.to_string(),
                "L2 cache" => self.cache_l2 = val.to_string(),
                "L3 cache" => self.cache_l3 = val.to_string(),
                "Flags" | "flags" => {
                    // Extract interesting ISA extensions
                    let interesting = [
                        "sse4_2", "avx", "avx2", "avx512f", "avx512vnni", "avx512bf16",
                        "amx_tile", "amx_bf16", "amx_int8",
                        "neon", "sve", "sve2", "bf16", "i8mm", "dotprod",
                    ];
                    let found: Vec<&str> = interesting.iter()
                        .filter(|ext| val.split_whitespace().any(|f| f == **ext))
                        .copied()
                        .collect();
                    self.isa_extensions = found.join(" ");
                }
                "Model name" => {
                    // lscpu often has a better model name than /proc/cpuinfo on ARM
                    if self.cpu_model.is_empty() || self.cpu_model == "?" {
                        self.cpu_model = val.to_string();
                    }
                }
                "CPU max MHz" | "CPU MHz" => {
                    if self.cpu_mhz == 0.0 {
                        self.cpu_mhz = val.parse().unwrap_or(0.0);
                    }
                }
                _ => {}
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn collect_dmidecode(&mut self) {
        // dmidecode needs root — user data runs as root on EC2
        let Ok(output) = Command::new("dmidecode").args(["-t", "17"]).output() else { return };
        let text = String::from_utf8_lossy(&output.stdout);

        let mut dimm_count = 0u32;
        let mut first_type = String::new();
        let mut first_speed = 0u32;

        // Parse each "Memory Device" block
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("Type:") && !trimmed.contains("Detail") && !trimmed.contains("Error") {
                let val = trimmed.split(':').nth(1).unwrap_or("").trim();
                if val != "Unknown" && val != "Other" && !val.is_empty() {
                    if first_type.is_empty() {
                        first_type = val.to_string();
                    }
                    dimm_count += 1;
                }
            }
            if trimmed.starts_with("Speed:") && !trimmed.contains("Unknown") && !trimmed.contains("Configured") {
                // "Speed: 4800 MT/s" or "Speed: 3200 MHz"
                let val = trimmed.split(':').nth(1).unwrap_or("").trim();
                let num_str: String = val.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(speed) = num_str.parse::<u32>() {
                    if first_speed == 0 {
                        first_speed = speed;
                    }
                }
            }
        }

        if !first_type.is_empty() {
            self.mem_type = first_type;
        }
        if first_speed > 0 {
            self.mem_speed_mt = first_speed;
        }
        if dimm_count > 0 {
            self.mem_channels = dimm_count;
        }
    }

    #[cfg(target_os = "linux")]
    fn collect_storage(&mut self) {
        let Ok(output) = Command::new("lsblk")
            .args(["-d", "-o", "NAME,TYPE,SIZE,ROTA", "--noheadings"])
            .output()
        else { return };
        let text = String::from_utf8_lossy(&output.stdout);

        // Find root disk (usually xvda, nvme0n1, or sda)
        for line in text.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 && parts[1] == "disk" {
                let rotational = parts[3] == "1";
                self.storage_type = if rotational { "HDD" } else { "SSD (NVMe/EBS)" }.to_string();
                self.storage_size = parts[2].to_string();
                break;
            }
        }

        // Refine: check if it's NVMe or EBS
        if let Ok(output) = Command::new("lsblk").args(["-d", "-o", "NAME,TRAN", "--noheadings"]).output() {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if parts[1].contains("nvme") {
                        self.storage_type = "NVMe SSD".to_string();
                    }
                    break;
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn collect_kernel(&mut self) {
        if let Ok(output) = Command::new("uname").arg("-r").output() {
            self.kernel_version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        }
    }
}
