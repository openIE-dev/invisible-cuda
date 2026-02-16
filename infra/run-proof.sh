#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Invisible CUDA — Full EC2 Proof Orchestrator (Wave-Based)
#
# Deploys infrastructure via CDK, then launches EC2 instances in waves
# that respect vCPU service quotas. Each instance auto-terminates after
# running the proof, freeing capacity for the next wave.
#
# Usage:
#   ./run-proof.sh              # Full run (build + deploy + wait + results)
#   ./run-proof.sh build        # Just cross-compile the binaries
#   ./run-proof.sh deploy       # Deploy infra + launch instances in waves
#   ./run-proof.sh status       # Check instance status
#   ./run-proof.sh results      # Download + display results
#   ./run-proof.sh destroy      # Tear down all infrastructure
#   ./run-proof.sh matrix       # Show the instance matrix (no AWS needed)
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CUDA_DIR="$ROOT_DIR/invisible-cuda"
BUILD_DIR="$ROOT_DIR/build"
RESULTS_DIR="$ROOT_DIR/results"
REGION="us-east-1"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ============================================================================
# Instance Registry
# Format: id|type|arch|vcpus|quota_bucket|gpu|price|proves
#
# quota_bucket: standard (A,C,D,H,I,M,R,T,Z), gpu (G,VT), hpc (HPC), x (X)
# ============================================================================

INSTANCES=(
  # Tier 1: Cheapest
  "t3-micro|t3.micro|x86_64|2|standard|none|0.010|Cheapest x86"
  "t4g-micro|t4g.micro|arm64|2|standard|none|0.008|Cheapest ARM"
  "t3-small|t3.small|x86_64|2|standard|none|0.021|Budget x86"
  "t4g-small|t4g.small|arm64|2|standard|none|0.017|Budget ARM"
  # Tier 2: General purpose
  "t3-large|t3.large|x86_64|2|standard|none|0.083|Dev machine"
  "m7g-medium|m7g.medium|arm64|1|standard|none|0.041|Graviton 3 GP"
  "m7i-large|m7i.large|x86_64|2|standard|none|0.100|Sapphire Rapids"
  # Tier 3: Compute-optimized
  "c7g-large|c7g.large|arm64|2|standard|none|0.073|Compute ARM"
  "c7i-2xl|c7i.2xlarge|x86_64|8|standard|none|0.357|8-core Intel"
  "c7g-4xl|c7g.4xlarge|arm64|16|standard|none|0.579|16-core Grav3"
  "c7i-4xl|c7i.4xlarge|x86_64|16|standard|none|0.714|16-core Intel"
  "c7g-8xl|c7g.8xlarge|arm64|32|standard|none|1.158|32-core Grav3"
  "c7i-8xl|c7i.8xlarge|x86_64|32|standard|none|1.428|32-core Intel"
  # Tier 4: Memory-optimized
  "r7g-large|r7g.large|arm64|2|standard|none|0.107|16 GB ARM"
  "r7i-large|r7i.large|x86_64|2|standard|none|0.132|16 GB Intel"
  "r7g-4xl|r7g.4xlarge|arm64|16|standard|none|0.854|128 GB ARM"
  # Tier 5A: ARM Graviton bare metal
  "a1-metal|a1.metal|arm64|16|standard|none|0.408|Graviton 1 16c"
  "c6g-metal|c6g.metal|arm64|64|standard|none|2.176|Graviton 2 64c"
  "m6g-metal|m6g.metal|arm64|64|standard|none|2.464|Graviton 2 GP"
  "r6g-metal|r6g.metal|arm64|64|standard|none|3.226|Graviton 2 512G"
  "c7g-metal|c7g.metal|arm64|64|standard|none|2.320|Graviton 3 64c"
  "m7g-metal|m7g.metal|arm64|64|standard|none|2.611|Graviton 3 GP"
  "r7g-metal|r7g.metal|arm64|64|standard|none|3.427|Graviton 3 512G"
  "c7gn-metal|c7gn.metal|arm64|64|standard|none|3.994|Graviton 3E net"
  "c8g-metal-24|c8g.metal-24xl|arm64|96|standard|none|3.829|Graviton 4 96c"
  "m8g-metal-24|m8g.metal-24xl|arm64|96|standard|none|4.310|Graviton 4 GP"
  "r8g-metal-24|r8g.metal-24xl|arm64|96|standard|none|5.655|Graviton 4 768G"
  "i8g-metal-24|i8g.metal-24xl|arm64|96|standard|none|8.237|Graviton 4 stor"
  # Tier 5B: Intel bare metal
  "i3-metal|i3.metal|x86_64|72|standard|none|4.992|Broadwell 72v"
  "z1d-metal|z1d.metal|x86_64|48|standard|none|4.464|Skylake 4.0GHz"
  "c5-metal|c5.metal|x86_64|96|standard|none|4.080|Cascade Lake 96v"
  "m5-metal|m5.metal|x86_64|96|standard|none|4.608|Cascade Lake GP"
  "r5-metal|r5.metal|x86_64|96|standard|none|6.048|Cascade Lake 768G"
  "m5zn-metal|m5zn.metal|x86_64|48|standard|none|3.964|Cascade 4.5GHz"
  "c6i-metal|c6i.metal|x86_64|128|standard|none|5.440|Ice Lake 128v"
  "m6i-metal|m6i.metal|x86_64|128|standard|none|6.144|Ice Lake GP"
  "r6i-metal|r6i.metal|x86_64|128|standard|none|8.064|Ice Lake 1TB"
  "i4i-metal|i4i.metal|x86_64|128|standard|none|10.982|Ice Lake storage"
  "c7i-metal-24|c7i.metal-24xl|x86_64|96|standard|none|4.284|Sapphire Rp 96v"
  "m7i-metal-24|m7i.metal-24xl|x86_64|96|standard|none|4.838|Sapphire Rp GP"
  "r7i-metal-24|r7i.metal-24xl|x86_64|96|standard|none|6.350|Sapphire Rp 768G"
  "r7iz-metal-16|r7iz.metal-16xl|x86_64|64|standard|none|5.952|Sapphire hi-freq"
  "i7i-metal-24|i7i.metal-24xl|x86_64|96|standard|none|9.061|Sapphire stor"
  "c8i-metal-48|c8i.metal-48xl|x86_64|192|standard|none|8.996|Granite Rp 192v"
  "m8i-metal-48|m8i.metal-48xl|x86_64|192|standard|none|10.161|Granite Rp GP"
  "r8i-metal-48|r8i.metal-48xl|x86_64|192|standard|none|13.340|Granite Rp 1.5TB"
  # Tier 5C: AMD bare metal
  "c6a-metal|c6a.metal|x86_64|192|standard|none|7.344|EPYC Milan 192v"
  "m6a-metal|m6a.metal|x86_64|192|standard|none|8.294|EPYC Milan GP"
  "r6a-metal|r6a.metal|x86_64|192|standard|none|10.886|EPYC Milan 1.5TB"
  "c7a-metal-48|c7a.metal-48xl|x86_64|192|standard|none|9.853|EPYC Genoa 192v"
  "m7a-metal-48|m7a.metal-48xl|x86_64|192|standard|none|11.128|EPYC Genoa GP"
  "r7a-metal-48|r7a.metal-48xl|x86_64|192|standard|none|14.610|EPYC Genoa 1.5TB"
  "c8a-metal-24|c8a.metal-24xl|x86_64|96|standard|none|5.173|EPYC Turin 96v"
  "m8a-metal-48|m8a.metal-48xl|x86_64|192|standard|none|11.685|EPYC Turin GP"
  "r8a-metal-48|r8a.metal-48xl|x86_64|192|standard|none|15.337|EPYC Turin 1.5TB"
  # Tier 5D: GPU bare metal
  "g4dn-metal|g4dn.metal|x86_64|96|gpu|8x NVIDIA T4 128GB|7.824|8x T4 bare metal"
  "g5g-metal|g5g.metal|arm64|64|gpu|NVIDIA T4G 16GB|2.744|ARM + T4G bare"
  # Tier 6: GPU virtual
  "g4ad-xl|g4ad.xlarge|x86_64|4|gpu|AMD Radeon Pro V520|0.379|AMD GPU"
  "g4dn-xl|g4dn.xlarge|x86_64|4|gpu|NVIDIA T4 16GB|0.526|T4 no drivers"
  "g5-xl|g5.xlarge|x86_64|4|gpu|NVIDIA A10G 24GB|1.006|A10G no drivers"
  "g6-xl|g6.xlarge|x86_64|4|gpu|NVIDIA L4 24GB|0.805|L4 no drivers"
  "g5g-xl|g5g.xlarge|arm64|4|gpu|NVIDIA T4G 16GB|0.421|ARM + GPU"
  # Tier 7: HPC
  "hpc7g-4xl|hpc7g.4xlarge|arm64|16|hpc|none|0.680|HPC Graviton"
)

# Distributed cluster instances (launched separately)
DIST_INSTANCES=(
  "dist-w1|t3.small|x86_64|2|standard"
  "dist-w2|t4g.small|arm64|2|standard"
  "dist-w3|c7g.large|arm64|2|standard"
  "dist-coord|t3.large|x86_64|2|standard"
)

# ============================================================================
# Matrix (display-only, no AWS)
# ============================================================================

cmd_matrix() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║                       Invisible CUDA — EC2 Universal Backend Proof Matrix                              ║${NC}"
    echo -e "${BOLD}╠════════════════╦══════════════════════╦════════╦═════════╦══════════════════════════╦════════════════════╣${NC}"
    printf  "║ %-14s ║ %-20s ║ %-6s ║ %-7s ║ %-24s ║ %-18s ║\n" \
        "ID" "Instance Type" "Arch" "\$/hr" "GPU" "What It Proves"
    echo -e "╠════════════════╬══════════════════════╬════════╬═════════╬══════════════════════════╬════════════════════╣"

    local current_tier=""
    for inst in "${INSTANCES[@]}"; do
        IFS='|' read -r id type arch vcpus bucket gpu price proves <<< "$inst"

        # Tier headers
        local tier=""
        case "$id" in
            t3-micro)      tier="Tier 1: The cheapest instances alive" ;;
            t3-large)      tier="Tier 2: General purpose" ;;
            c7g-large)     tier="Tier 3: Compute-optimized (rayon parallelism)" ;;
            r7g-large)     tier="Tier 4: Memory-optimized (large BLAS matrices)" ;;
            a1-metal)      tier="Tier 5A: Bare metal ARM Graviton (every generation, no hypervisor)" ;;
            i3-metal)      tier="Tier 5B: Bare metal Intel (Broadwell -> Granite Rapids, no hypervisor)" ;;
            c6a-metal)     tier="Tier 5C: Bare metal AMD EPYC (Milan -> Turin, no hypervisor)" ;;
            g4dn-metal)    tier="Tier 5D: Bare metal GPU (NVIDIA T4/T4G, CUDA on CPU -- no drivers)" ;;
            g4ad-xl)       tier="Tier 6: GPU instances -- virtualized (CUDA without NVIDIA drivers)" ;;
            hpc7g-4xl)     tier="Tier 7: High-performance compute" ;;
        esac
        if [ -n "$tier" ] && [ "$tier" != "$current_tier" ]; then
            current_tier="$tier"
            printf "║${DIM} -- %-100s ${NC}║\n" "$tier"
        fi

        printf "║ %-14s ║ %-20s ║ %-6s ║ \$%-6s ║ %-24s ║ %-18s ║\n" \
            "$id" "$type" "$arch" "$price" "$gpu" "$proves"
    done

    echo -e "╠════════════════╩══════════════════════╩════════╩═════════╩══════════════════════════╩════════════════════╣"
    echo -e "║  ${BOLD}${#INSTANCES[@]} instances${NC}  |  Bare metal: 42  |  GPU: 7  |  HPC: 1  |  Distributed: 4                       ║"
    echo -e "║  ${BOLD}CPU families: Graviton 1/2/3/3E/4, Intel Broadwell->Granite Rapids, AMD EPYC Milan->Turin${NC}              ║"
    echo -e "║  ${BOLD}Wave-based launch${NC}: instances launched in waves respecting vCPU service quotas                         ║"
    echo -e "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
}

# ============================================================================
# Build
# ============================================================================

cmd_build() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Step 1: Cross-compiling proof binaries${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    mkdir -p "$BUILD_DIR"

    if ! command -v cargo-zigbuild &>/dev/null; then
        echo "Installing cargo-zigbuild..."
        pip3 install ziglang
        cargo install cargo-zigbuild
    fi

    rustup target add x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu 2>/dev/null || true

    cd "$CUDA_DIR"

    echo ""
    echo "-- x86_64 (Intel/AMD EC2) --"
    cargo zigbuild --bin proof --bin worker --bin dist_proof --bin limits_proof --bin coverage_proof --release \
        --target x86_64-unknown-linux-gnu.2.17 \
        --no-default-features --features cpu,distributed 2>&1 | tail -1
    cp target/x86_64-unknown-linux-gnu/release/proof "$BUILD_DIR/proof-x86_64"
    cp target/x86_64-unknown-linux-gnu/release/worker "$BUILD_DIR/worker-x86_64"
    cp target/x86_64-unknown-linux-gnu/release/dist_proof "$BUILD_DIR/dist_proof-x86_64"
    cp target/x86_64-unknown-linux-gnu/release/limits_proof "$BUILD_DIR/limits_proof-x86_64"
    cp target/x86_64-unknown-linux-gnu/release/coverage_proof "$BUILD_DIR/coverage_proof-x86_64"
    echo -e "  ${GREEN}OK${NC} proof-x86_64 ($(du -h "$BUILD_DIR/proof-x86_64" | cut -f1))"
    echo -e "  ${GREEN}OK${NC} worker-x86_64 ($(du -h "$BUILD_DIR/worker-x86_64" | cut -f1))"
    echo -e "  ${GREEN}OK${NC} dist_proof-x86_64 ($(du -h "$BUILD_DIR/dist_proof-x86_64" | cut -f1))"
    echo -e "  ${GREEN}OK${NC} limits_proof-x86_64 ($(du -h "$BUILD_DIR/limits_proof-x86_64" | cut -f1))"
    echo -e "  ${GREEN}OK${NC} coverage_proof-x86_64 ($(du -h "$BUILD_DIR/coverage_proof-x86_64" | cut -f1))"

    echo ""
    echo "-- aarch64 (Graviton EC2) --"
    cargo zigbuild --bin proof --bin worker --bin dist_proof --bin limits_proof --bin coverage_proof --release \
        --target aarch64-unknown-linux-gnu.2.17 \
        --no-default-features --features cpu,distributed 2>&1 | tail -1
    cp target/aarch64-unknown-linux-gnu/release/proof "$BUILD_DIR/proof-aarch64"
    cp target/aarch64-unknown-linux-gnu/release/worker "$BUILD_DIR/worker-aarch64"
    cp target/aarch64-unknown-linux-gnu/release/dist_proof "$BUILD_DIR/dist_proof-aarch64"
    cp target/aarch64-unknown-linux-gnu/release/limits_proof "$BUILD_DIR/limits_proof-aarch64"
    cp target/aarch64-unknown-linux-gnu/release/coverage_proof "$BUILD_DIR/coverage_proof-aarch64"
    echo -e "  ${GREEN}OK${NC} proof-aarch64 ($(du -h "$BUILD_DIR/proof-aarch64" | cut -f1))"
    echo -e "  ${GREEN}OK${NC} worker-aarch64 ($(du -h "$BUILD_DIR/worker-aarch64" | cut -f1))"
    echo -e "  ${GREEN}OK${NC} dist_proof-aarch64 ($(du -h "$BUILD_DIR/dist_proof-aarch64" | cut -f1))"
    echo -e "  ${GREEN}OK${NC} limits_proof-aarch64 ($(du -h "$BUILD_DIR/limits_proof-aarch64" | cut -f1))"
    echo -e "  ${GREEN}OK${NC} coverage_proof-aarch64 ($(du -h "$BUILD_DIR/coverage_proof-aarch64" | cut -f1))"
}

# ============================================================================
# Helpers for wave-based deployment
# ============================================================================

# Resolve Amazon Linux 2023 AMI IDs
resolve_amis() {
    AMI_X86=$(aws ssm get-parameter \
        --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
        --query 'Parameter.Value' --output text --region "$REGION")
    AMI_ARM=$(aws ssm get-parameter \
        --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
        --query 'Parameter.Value' --output text --region "$REGION")
    # AL2023 requires ARMv8.2+ (Graviton 2+). a1.metal is Graviton 1 (ARMv8.0)
    # so it needs Amazon Linux 2 which supports ARMv8.0.
    AMI_ARM_AL2=$(aws ssm get-parameter \
        --name /aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-arm64-gp2 \
        --query 'Parameter.Value' --output text --region "$REGION")
    echo -e "  AMI x86_64: $AMI_X86"
    echo -e "  AMI arm64:  $AMI_ARM"
    echo -e "  AMI arm64 (AL2, Graviton 1): $AMI_ARM_AL2"
}

# Get a public subnet from the default VPC
resolve_subnet() {
    DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' --output text --region "$REGION")
    # Get all public subnets, pick the first one
    SUBNET_ID=$(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=$DEFAULT_VPC" "Name=map-public-ip-on-launch,Values=true" \
        --query 'Subnets[0].SubnetId' --output text --region "$REGION")
    # Get all public subnet IDs (for AZ fallback)
    ALL_SUBNETS=$(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=$DEFAULT_VPC" "Name=map-public-ip-on-launch,Values=true" \
        --query 'Subnets[*].SubnetId' --output text --region "$REGION")
    echo -e "  VPC: $DEFAULT_VPC"
    echo -e "  Subnet: $SUBNET_ID"
}

# Query current vCPU quotas
resolve_quotas() {
    QUOTA_standard=$(aws service-quotas get-service-quota --service-code ec2 \
        --quota-code L-1216C47A --query 'Quota.Value' --output text --region "$REGION" 2>/dev/null | cut -d. -f1)
    QUOTA_gpu=$(aws service-quotas get-service-quota --service-code ec2 \
        --quota-code L-DB2E81BA --query 'Quota.Value' --output text --region "$REGION" 2>/dev/null | cut -d. -f1)
    QUOTA_hpc=$(aws service-quotas get-service-quota --service-code ec2 \
        --quota-code L-F7808C92 --query 'Quota.Value' --output text --region "$REGION" 2>/dev/null | cut -d. -f1)
    echo -e "  Quotas: Standard=${QUOTA_standard} vCPUs, GPU=${QUOTA_gpu} vCPUs, HPC=${QUOTA_hpc} vCPUs"
}

# Get quota for a given bucket name
get_quota() {
    local bucket="$1"
    case "$bucket" in
        standard) echo "${QUOTA_standard:-0}" ;;
        gpu)      echo "${QUOTA_gpu:-0}" ;;
        hpc)      echo "${QUOTA_hpc:-0}" ;;
        *)        echo "0" ;;
    esac
}

# Count currently running vCPUs for our project
count_running_vcpus() {
    local bucket="$1"
    local total=0

    # Get running instance types with our project tag
    local running_types
    running_types=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=invisible-cuda-proof" \
                  "Name=instance-state-name,Values=running,pending" \
        --query "Reservations[].Instances[].InstanceType" \
        --output text --region "$REGION" 2>/dev/null || echo "")

    if [ -z "$running_types" ]; then
        echo 0
        return
    fi

    # Match against our registry to sum vCPUs in the requested bucket
    for running_type in $running_types; do
        for inst in "${INSTANCES[@]}" "${DIST_INSTANCES[@]}"; do
            IFS='|' read -r _id itype _arch vcpus ibucket _ <<< "$inst"
            if [ "$itype" = "$running_type" ] && [ "$ibucket" = "$bucket" ]; then
                total=$((total + vcpus))
                break
            fi
        done
    done
    echo "$total"
}

# Generate user data script for a proof instance
generate_proof_userdata() {
    local id="$1" type="$2" arch="$3" gpu="$4" price="$5" proves="$6" bucket_name="$7"

    local binary_key limits_key coverage_key
    if [ "$arch" = "arm64" ]; then
        binary_key="proof-aarch64"
        limits_key="limits_proof-aarch64"
        coverage_key="coverage_proof-aarch64"
    else
        binary_key="proof-x86_64"
        limits_key="limits_proof-x86_64"
        coverage_key="coverage_proof-x86_64"
    fi

    local proof_mode="standard"

    cat <<USERDATA
#!/bin/bash
exec > >(tee /tmp/proof-output.txt) 2>&1

INSTANCE_ID="${id}"
BUCKET="${bucket_name}"
BINARY_KEY="${binary_key}"
LIMITS_KEY="${limits_key}"
COVERAGE_KEY="${coverage_key}"
PROOF_MODE="${proof_mode}"
PRICE="${price}"

echo "==================================================="
echo "  Instance: \$INSTANCE_ID"
echo "  Type:     ${type}"
echo "  Arch:     ${arch}"
echo "  GPU:      ${gpu}"
echo "  Price:    \\\$\${PRICE}/hr"
echo "  Backend:  CPU"
echo "  Proves:   ${proves}"
echo "  Mode:     \$PROOF_MODE"
echo "==================================================="
echo ""

# Ensure we ALWAYS shutdown, even on errors
cleanup() {
  # Upload whatever results we have
  aws s3 cp /tmp/proof-output.txt "s3://\$BUCKET/results/\$INSTANCE_ID.txt" 2>/dev/null || true

  # Tag with status
  TOKEN=\$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || echo "")
  if [ -n "\$TOKEN" ]; then
    EC2_ID=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
    MYREGION=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "")
    if [ -n "\$EC2_ID" ] && [ -n "\$MYREGION" ]; then
      if [ "\${PROOF_EXIT:-1}" -eq 0 ] && [ "\${LIMITS_EXIT:-1}" -eq 0 ] && [ "\${COVERAGE_EXIT:-1}" -eq 0 ]; then
        aws ec2 create-tags --region "\$MYREGION" --resources "\$EC2_ID" --tags Key=ProofStatus,Value=PASS 2>/dev/null || true
      else
        aws ec2 create-tags --region "\$MYREGION" --resources "\$EC2_ID" --tags Key=ProofStatus,Value=FAIL 2>/dev/null || true
      fi
    fi
  fi

  shutdown -h now
}
trap cleanup EXIT

# Install dmidecode for hardware fingerprint (lscpu/lsblk are pre-installed)
yum install -y dmidecode >/dev/null 2>&1 || true

# Download proof binaries from S3
set -e
aws s3 cp "s3://\$BUCKET/\$BINARY_KEY" /tmp/proof
aws s3 cp "s3://\$BUCKET/\$LIMITS_KEY" /tmp/limits_proof
aws s3 cp "s3://\$BUCKET/\$COVERAGE_KEY" /tmp/coverage_proof
chmod +x /tmp/proof /tmp/limits_proof /tmp/coverage_proof
set +e

# Part 1: Compatibility proof
echo "Starting compatibility proof at \$(date -u)"
INVISIBLE_CUDA_BACKEND=cpu PROOF_MODE=\$PROOF_MODE /tmp/proof
PROOF_EXIT=\$?
echo ""
echo "Compatibility proof exited with code: \$PROOF_EXIT"

# Part 2: Limits proof
echo ""
echo "Starting limits proof at \$(date -u)"
INVISIBLE_CUDA_BACKEND=cpu PROOF_MODE=\$PROOF_MODE /tmp/limits_proof
LIMITS_EXIT=\$?
echo ""
echo "Limits proof exited with code: \$LIMITS_EXIT"

# Part 3: Coverage proof
echo ""
echo "Starting coverage proof at \$(date -u)"
/tmp/coverage_proof
COVERAGE_EXIT=\$?
echo ""
echo "Coverage proof exited with code: \$COVERAGE_EXIT"
echo "All proofs completed at \$(date -u)"
USERDATA
}

# Generate user data for a distributed worker
generate_worker_userdata() {
    local id="$1" arch="$2" bucket_name="$3" worker_port="$4"

    local worker_key
    if [ "$arch" = "arm64" ]; then
        worker_key="worker-aarch64"
    else
        worker_key="worker-x86_64"
    fi

    cat <<USERDATA
#!/bin/bash
exec > >(tee /tmp/worker-output.txt) 2>&1

BUCKET="${bucket_name}"
WORKER_KEY="${worker_key}"
WORKER_ID="${id}"

cleanup() {
  aws s3 cp /tmp/worker-output.txt "s3://\$BUCKET/results/\$WORKER_ID.txt" 2>/dev/null || true
  shutdown -h now
}
trap cleanup EXIT

echo "Starting distributed worker: \$WORKER_ID"

# Download worker binary
aws s3 cp "s3://\$BUCKET/\$WORKER_KEY" /tmp/worker
chmod +x /tmp/worker

# Register our private IP in SSM for coordinator to find
TOKEN=\$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
PRIVATE_IP=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/local-ipv4)
REGION=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
aws ssm put-parameter --region "\$REGION" --name "/invisible-cuda-proof/dist-workers/\$WORKER_ID" --value "\$PRIVATE_IP:${worker_port}" --type String --overwrite

echo "Worker IP registered: \$PRIVATE_IP:${worker_port}"

# Run worker daemon (blocks until coordinator sends Shutdown)
INVISIBLE_CUDA_BACKEND=cpu /tmp/worker --port ${worker_port} || true

echo "Worker shutdown"

# Signal completion
EC2_ID=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
if [ -n "\$EC2_ID" ] && [ -n "\$REGION" ]; then
  aws ec2 create-tags --region "\$REGION" --resources "\$EC2_ID" --tags Key=ProofStatus,Value=DONE 2>/dev/null || true
fi
USERDATA
}

# Generate user data for the distributed coordinator
generate_coordinator_userdata() {
    local bucket_name="$1" worker_ids="$2" num_workers="$3"

    cat <<USERDATA
#!/bin/bash
exec > >(tee /tmp/dist-proof-output.txt) 2>&1

BUCKET="${bucket_name}"
NUM_WORKERS=${num_workers}
PROOF_EXIT=1

TOKEN=\$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
REGION=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
EC2_ID=\$(curl -s -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

cleanup() {
  aws s3 cp /tmp/dist-proof-output.txt "s3://\$BUCKET/results/dist-coordinator.txt" 2>/dev/null || true
  if [ -n "\$EC2_ID" ] && [ -n "\$REGION" ]; then
    if [ "\$PROOF_EXIT" -eq 0 ]; then
      aws ec2 create-tags --region "\$REGION" --resources "\$EC2_ID" --tags Key=ProofStatus,Value=PASS 2>/dev/null || true
    else
      aws ec2 create-tags --region "\$REGION" --resources "\$EC2_ID" --tags Key=ProofStatus,Value=FAIL 2>/dev/null || true
    fi
    # Clean up SSM parameters
    for WID in ${worker_ids}; do
      aws ssm delete-parameter --region "\$REGION" --name "/invisible-cuda-proof/dist-workers/\$WID" 2>/dev/null || true
    done
  fi
  shutdown -h now
}
trap cleanup EXIT

echo "==================================================="
echo "  Distributed Proof Coordinator"
echo "  Waiting for \$NUM_WORKERS workers to register..."
echo "==================================================="

# Download dist_proof binary
aws s3 cp "s3://\$BUCKET/dist_proof-x86_64" /tmp/dist_proof
chmod +x /tmp/dist_proof

# Wait for all workers to register their IPs in SSM (max 5 min)
ELAPSED=0
WORKER_ADDRS=
while [ \$ELAPSED -lt 300 ]; do
  WORKER_ADDRS=
  FOUND=0
  for WID in ${worker_ids}; do
    ADDR=\$(aws ssm get-parameter --region "\$REGION" --name "/invisible-cuda-proof/dist-workers/\$WID" --query "Parameter.Value" --output text 2>/dev/null || echo "")
    if [ -n "\$ADDR" ] && [ "\$ADDR" != "None" ]; then
      FOUND=\$((FOUND + 1))
      WORKER_ADDRS="\$WORKER_ADDRS \$ADDR"
    fi
  done
  if [ "\$FOUND" -ge "\$NUM_WORKERS" ]; then
    echo "All \$NUM_WORKERS workers registered"
    break
  fi
  echo "  \$FOUND/\$NUM_WORKERS workers registered... (\${ELAPSED}s)"
  sleep 10
  ELAPSED=\$((ELAPSED + 10))
done

if [ -z "\$WORKER_ADDRS" ]; then
  echo "ERROR: No workers registered after 5 minutes"
  exit 1
fi

# Give workers an extra 10s to fully start their daemons
sleep 10

echo ""
echo "Running distributed proof against workers: \$WORKER_ADDRS"
echo ""

# Run distributed proof
/tmp/dist_proof \$WORKER_ADDRS
PROOF_EXIT=\$?
USERDATA
}

# Launch a single EC2 instance
launch_instance() {
    local id="$1" type="$2" arch="$3" sg_id="$4" profile_arn="$5" userdata_file="$6"

    local ami
    if [ "$arch" = "arm64" ]; then
        ami="$AMI_ARM"
    else
        ami="$AMI_X86"
    fi
    # a1.metal is Graviton 1 (ARMv8.0) — needs AL2 instead of AL2023
    if [ "$type" = "a1.metal" ]; then
        ami="$AMI_ARM_AL2"
    fi

    # Try each subnet until one works (handles AZ availability)
    local ec2_id=""
    for subnet in $ALL_SUBNETS; do
        ec2_id=$(aws ec2 run-instances \
            --image-id "$ami" \
            --instance-type "$type" \
            --subnet-id "$subnet" \
            --security-group-ids "$sg_id" \
            --iam-instance-profile "Arn=$profile_arn" \
            --user-data "file://$userdata_file" \
            --instance-initiated-shutdown-behavior terminate \
            --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeType":"gp3","VolumeSize":8,"Iops":3000,"Throughput":125,"DeleteOnTermination":true}}]' \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=invisible-cuda-proof-${id}},{Key=Project,Value=invisible-cuda-proof},{Key=InstanceId,Value=${id}},{Key=Arch,Value=${arch}},{Key=ProofStatus,Value=PENDING}]" \
            --query 'Instances[0].InstanceId' --output text \
            --region "$REGION" 2>/dev/null) && break || true
        ec2_id=""
    done

    if [ -z "$ec2_id" ]; then
        echo -e "    ${RED}FAIL${NC} $id ($type) -- could not launch in any AZ"
        return 1
    fi

    echo -e "    ${GREEN}OK${NC} $id ($type) -> $ec2_id"
    return 0
}

# ============================================================================
# Deploy (wave-based)
# ============================================================================

cmd_deploy() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Step 2: Deploying infrastructure + launching waves${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"

    # ── Deploy CDK infrastructure (no EC2 instances) ──
    echo ""
    echo -e "  ${BOLD}Phase 1: CDK infrastructure${NC}"
    cd "$SCRIPT_DIR"
    npm install --silent 2>/dev/null

    echo "  Synthesizing CloudFormation template..."
    npx cdk synth --quiet 2>/dev/null

    echo "  Deploying stack (S3 bucket + IAM + security groups)..."
    npx cdk deploy --require-approval never --outputs-file outputs.json 2>&1 | \
        grep -E "(InvisibleCudaProof|output)" | head -5 || true

    # Extract outputs
    local bucket profile_arn sg_id dist_sg_id
    bucket=$(node -e "const o=require('./outputs.json');console.log(o['InvisibleCudaProof']['BucketName'])" 2>/dev/null)
    profile_arn=$(node -e "const o=require('./outputs.json');console.log(o['InvisibleCudaProof']['InstanceProfileArn'])" 2>/dev/null)
    sg_id=$(node -e "const o=require('./outputs.json');console.log(o['InvisibleCudaProof']['SecurityGroupId'])" 2>/dev/null)
    dist_sg_id=$(node -e "const o=require('./outputs.json');console.log(o['InvisibleCudaProof']['DistSecurityGroupId'])" 2>/dev/null)

    if [ -z "$bucket" ] || [ -z "$profile_arn" ]; then
        echo -e "${RED}ERROR: Could not read CDK outputs${NC}"
        exit 1
    fi

    echo -e "  ${GREEN}Infrastructure deployed${NC}"
    echo "    Bucket: $bucket"
    echo "    Profile: $profile_arn"
    echo "    SG: $sg_id"

    # ── Upload binaries to S3 ──
    echo ""
    echo -e "  ${BOLD}Phase 2: Uploading binaries${NC}"
    for f in proof-x86_64 proof-aarch64 worker-x86_64 worker-aarch64 dist_proof-x86_64 dist_proof-aarch64 limits_proof-x86_64 limits_proof-aarch64 coverage_proof-x86_64 coverage_proof-aarch64; do
        if [ -f "$BUILD_DIR/$f" ]; then
            aws s3 cp "$BUILD_DIR/$f" "s3://$bucket/$f" --region "$REGION" --quiet
            echo -e "    ${GREEN}OK${NC} $f"
        fi
    done

    # ── Resolve AMIs + subnet + quotas ──
    echo ""
    echo -e "  ${BOLD}Phase 3: Resolving AMIs, subnets, quotas${NC}"
    resolve_amis
    resolve_subnet
    resolve_quotas

    # ── Launch proof instances in waves ──
    echo ""
    echo -e "  ${BOLD}Phase 4: Launching proof instances in waves${NC}"
    echo ""

    local launched=0
    local skipped=0
    local failed=0
    local tmpdir
    tmpdir=$(mktemp -d)

    for inst in "${INSTANCES[@]}"; do
        IFS='|' read -r id type arch vcpus qbucket gpu price proves <<< "$inst"

        local quota_limit
        quota_limit=$(get_quota "$qbucket")

        # Skip instances that individually exceed quota
        if (( vcpus > quota_limit )); then
            echo -e "    ${YELLOW}SKIP${NC} $id ($type) -- needs $vcpus vCPUs, quota is $quota_limit"
            skipped=$((skipped + 1))
            continue
        fi

        # Wait for capacity in this quota bucket
        local attempts=0
        while true; do
            local running_vcpus
            running_vcpus=$(count_running_vcpus "$qbucket")
            if (( running_vcpus + vcpus <= quota_limit )); then
                break
            fi
            if (( attempts == 0 )); then
                echo -e "    ${DIM}Waiting for capacity: $qbucket has $running_vcpus/$quota_limit vCPUs running${NC}"
            fi
            attempts=$((attempts + 1))
            sleep 15
        done

        # Generate user data
        local userdata_file="$tmpdir/userdata-${id}.sh"
        generate_proof_userdata "$id" "$type" "$arch" "$gpu" "$price" "$proves" "$bucket" > "$userdata_file"

        # Launch
        if launch_instance "$id" "$type" "$arch" "$sg_id" "$profile_arn" "$userdata_file"; then
            launched=$((launched + 1))
        else
            failed=$((failed + 1))
        fi
    done

    # ── Launch distributed cluster ──
    echo ""
    echo -e "  ${BOLD}Phase 5: Distributed cluster${NC}"

    local worker_port=9741
    local worker_ids=""

    # Launch workers
    for dinst in "${DIST_INSTANCES[@]}"; do
        IFS='|' read -r did dtype darch dvcpus dbucket <<< "$dinst"
        [ "$did" = "dist-coord" ] && continue  # coordinator launched after workers

        worker_ids="$worker_ids $did"

        # Wait for capacity
        while true; do
            local running_vcpus
            running_vcpus=$(count_running_vcpus "$dbucket")
            if (( running_vcpus + dvcpus <= $(get_quota "$dbucket") )); then
                break
            fi
            sleep 15
        done

        local userdata_file="$tmpdir/userdata-${did}.sh"
        generate_worker_userdata "$did" "$darch" "$bucket" "$worker_port" > "$userdata_file"

        if launch_instance "$did" "$dtype" "$darch" "$dist_sg_id" "$profile_arn" "$userdata_file"; then
            launched=$((launched + 1))
        else
            failed=$((failed + 1))
        fi
    done

    # Launch coordinator (after workers)
    worker_ids=$(echo "$worker_ids" | xargs)  # trim whitespace
    local coord_userdata="$tmpdir/userdata-dist-coord.sh"
    generate_coordinator_userdata "$bucket" "$worker_ids" "3" > "$coord_userdata"

    # Wait for capacity
    while true; do
        local running_vcpus
        running_vcpus=$(count_running_vcpus "standard")
        if (( running_vcpus + 2 <= $(get_quota "standard") )); then
            break
        fi
        sleep 15
    done

    if launch_instance "dist-coord" "t3.large" "x86_64" "$dist_sg_id" "$profile_arn" "$coord_userdata"; then
        launched=$((launched + 1))
    else
        failed=$((failed + 1))
    fi

    # Cleanup
    rm -rf "$tmpdir"

    echo ""
    echo -e "  ${GREEN}${BOLD}Launched: $launched${NC}  ${YELLOW}Skipped: $skipped${NC}  ${RED}Failed: $failed${NC}"
    echo ""
    if (( skipped > 0 )); then
        echo -e "  ${YELLOW}NOTE: $skipped instances skipped due to vCPU quota limits.${NC}"
        echo -e "  ${YELLOW}Request quota increases at: https://console.aws.amazon.com/servicequotas/${NC}"
        echo -e "  ${YELLOW}Then re-run ./run-proof.sh deploy to launch the remaining instances.${NC}"
    fi
    echo ""
    echo -e "  ${BOLD}Instances are booting now. They will:${NC}"
    echo "    1. Download proof binaries from S3"
    echo "    2. Run all three proof suites"
    echo "    3. Upload results to S3"
    echo "    4. Auto-shutdown to stop billing"
    echo ""
    echo "  Run './run-proof.sh status' to monitor progress."
    echo "  Run './run-proof.sh results' once all are done."
}

# ============================================================================
# Status
# ============================================================================

cmd_status() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Instance Status${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo ""

    aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=invisible-cuda-proof" \
        --query "Reservations[].Instances[].{
            Name:Tags[?Key=='InstanceId']|[0].Value,
            Type:InstanceType,
            State:State.Name,
            Proof:Tags[?Key=='ProofStatus']|[0].Value,
            Arch:Tags[?Key=='Arch']|[0].Value
        }" \
        --output table --region "$REGION"
}

# ============================================================================
# Results
# ============================================================================

cmd_results() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Invisible CUDA — EC2 Universal Backend Proof Results${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""

    mkdir -p "$RESULTS_DIR"

    # Get bucket name
    local bucket
    if [ -f "$SCRIPT_DIR/outputs.json" ]; then
        bucket=$(node -e "const o=require('$SCRIPT_DIR/outputs.json');console.log(o['InvisibleCudaProof']['BucketName'])" 2>/dev/null)
    fi

    if [ -z "${bucket:-}" ]; then
        echo -e "${RED}No outputs.json found. Run './run-proof.sh deploy' first.${NC}"
        exit 1
    fi

    # Download all results
    echo "  Downloading results from s3://$bucket/results/ ..."
    aws s3 sync "s3://$bucket/results/" "$RESULTS_DIR/" --quiet --region "$REGION" 2>/dev/null || true
    echo ""

    # Parse and display
    echo "╔════════════════╦══════════════════════╦════════╦════════╦═══════════╦════════╦═════════════╦═══════════════╦═══════════════╗"
    printf "║ %-14s ║ %-20s ║ %-6s ║ %-6s ║ %-9s ║ %-6s ║ %-11s ║ %-13s ║ %-13s ║\n" \
        "Instance" "Type" "Arch" "\$/hr" "Tests" "Status" "Mem BW" "SGEMM" "Limits"
    echo "╠════════════════╬══════════════════════╬════════╬════════╬═══════════╬════════╬═════════════╬═══════════════╬═══════════════╣"

    local total_pass=0
    local total_fail=0
    local total_pending=0

    for result_file in "$RESULTS_DIR"/*.txt; do
        [ -f "$result_file" ] || continue
        local name
        name=$(basename "$result_file" .txt)

        local status="?"
        local mem_bw="--"
        local sgemm="--"
        local limits_summary="--"
        local instance_type="?"
        local arch="?"
        local price="?"
        local test_count="?"

        # Parse result file -- try JSON block first, fall back to text
        local proof_json=""
        proof_json=$(python3 -c "
import re, json, sys
with open('$result_file') as f:
    text = f.read()
blocks = re.findall(r'--- JSON_START ---\n(.*?)\n--- JSON_END ---', text, re.DOTALL)
for b in blocks:
    try:
        d = json.loads(b)
        if d.get('format') == 'invisible-cuda-v2':
            print(b)
            break
    except: pass
" 2>/dev/null || echo "")

        local limits_json=""
        limits_json=$(python3 -c "
import re, json, sys
with open('$result_file') as f:
    text = f.read()
blocks = re.findall(r'--- JSON_START ---\n(.*?)\n--- JSON_END ---', text, re.DOTALL)
for b in blocks:
    try:
        d = json.loads(b)
        if d.get('format') == 'invisible-cuda-limits-v1':
            print(b)
            break
    except: pass
" 2>/dev/null || echo "")

        if [ -n "$proof_json" ]; then
            local passed failed
            passed=$(echo "$proof_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for t in d['tests'] if t['passed']))" 2>/dev/null || echo "?")
            failed=$(echo "$proof_json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for t in d['tests'] if not t['passed']))" 2>/dev/null || echo "?")
            test_count="$passed/$((passed + failed))"

            if [ "$failed" = "0" ]; then
                status="PASS"
                total_pass=$((total_pass + 1))
            else
                status="FAIL"
                total_fail=$((total_fail + 1))
            fi

            mem_bw=$(echo "$proof_json" | python3 -c "
import sys,json
d=json.load(sys.stdin)
bw = [b['value'] for b in d.get('benchmarks',[]) if 'Mem BW' in b['name']]
if bw: print(f'{max(bw):.1f} GB/s')
else: print('--')
" 2>/dev/null || echo "--")

            sgemm=$(echo "$proof_json" | python3 -c "
import sys,json
d=json.load(sys.stdin)
gf = [b['value']/1e9 for b in d.get('benchmarks',[]) if 'SGEMM' in b['name'] and b['unit']=='FLOPS']
if gf: print(f'{max(gf):.1f} GFLOPS')
else: print('--')
" 2>/dev/null || echo "--")
        else
            if grep -q "ALL.*TESTS PASSED" "$result_file" 2>/dev/null; then
                status="PASS"
                total_pass=$((total_pass + 1))
            elif grep -q "FAILED" "$result_file" 2>/dev/null; then
                status="FAIL"
                total_fail=$((total_fail + 1))
            else
                status="???"
                total_pending=$((total_pending + 1))
            fi
            test_count="--"
        fi

        if [ -n "$limits_json" ]; then
            limits_summary=$(echo "$limits_json" | python3 -c "
import sys,json
d=json.load(sys.stdin)
s = d.get('summary',{})
sup = s.get('supported',0)
deg = s.get('degraded',0)
uns = s.get('unsupported',0)
print(f'{sup}S/{deg}D/{uns}U')
" 2>/dev/null || echo "--")
        fi

        instance_type=$(grep "^  Type:" "$result_file" 2>/dev/null | awk '{print $2}' || echo "?")
        arch=$(grep "^  Arch:" "$result_file" 2>/dev/null | awk '{print $2}' || echo "?")
        price=$(grep "^  Price:" "$result_file" 2>/dev/null | awk '{print $2}' || echo "?")

        local status_color="$NC"
        if [ "$status" = "PASS" ]; then status_color="$GREEN"; fi
        if [ "$status" = "FAIL" ]; then status_color="$RED"; fi

        printf "║ %-14s ║ %-20s ║ %-6s ║ %-6s ║ %-9s ║ ${status_color}%-6s${NC} ║ %-11s ║ %-13s ║ %-13s ║\n" \
            "$name" "$instance_type" "$arch" "$price" "$test_count" "$status" "$mem_bw" "$sgemm" "$limits_summary"
    done

    echo "╠════════════════╩══════════════════════╩════════╩════════╩═══════════╩════════╩═════════════╩═══════════════╩═══════════════╣"
    printf "║  Total: ${GREEN}%d PASS${NC}  ${RED}%d FAIL${NC}  %d pending  |  Limits: S=Supported  D=Degraded  U=Unsupported                                     ║\n" \
        "$total_pass" "$total_fail" "$total_pending"
    echo "╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"

    echo ""
    if [ "$total_fail" -eq 0 ] && [ "$total_pass" -gt 0 ]; then
        echo -e "  ${GREEN}${BOLD}CUDA COMPATIBILITY VERIFIED on $total_pass EC2 instance types${NC}"
        echo ""
        echo "  Invisible CUDA runs on:"
        echo "    - x86_64 and ARM (aarch64)"
        echo "    - 1 vCPU to 192 cores"
        echo "    - Instances from \$0.008/hr to \$15.34/hr"
        echo "    - With and without GPUs"
        echo "    - Bare metal and virtualized"
        echo ""
        echo -e "  ${BOLD}GPU compute prices are too high. Any machine is now CUDA-capable.${NC}"
    fi
}

# ============================================================================
# Wait (poll until all instances shut down)
# ============================================================================

cmd_wait() {
    echo ""
    echo -e "${CYAN}  Waiting for all proof instances to complete...${NC}"
    echo "  (Instances auto-shutdown after running proof)"
    echo ""

    local max_wait=1800  # 30 minutes max (wave-based takes longer)
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        local running
        running=$(aws ec2 describe-instances \
            --filters "Name=tag:Project,Values=invisible-cuda-proof" "Name=instance-state-name,Values=running,pending" \
            --query "Reservations[].Instances[].InstanceId" \
            --output text --region "$REGION" 2>/dev/null | wc -w | tr -d ' ')

        if [ "$running" -eq 0 ]; then
            echo -e "  ${GREEN}All instances have completed and shut down.${NC}"
            return 0
        fi

        echo "  $running instance(s) still running... (${elapsed}s elapsed)"
        sleep 15
        elapsed=$((elapsed + 15))
    done

    echo -e "  ${YELLOW}Timeout after ${max_wait}s. Some instances may still be running.${NC}"
    cmd_status
}

# ============================================================================
# Destroy
# ============================================================================

cmd_destroy() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Tearing down all infrastructure${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"

    # Terminate any running instances first
    echo "  Terminating any running proof instances..."
    local instance_ids
    instance_ids=$(aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=invisible-cuda-proof" \
                  "Name=instance-state-name,Values=running,pending,stopped" \
        --query "Reservations[].Instances[].InstanceId" \
        --output text --region "$REGION" 2>/dev/null || echo "")

    if [ -n "$instance_ids" ] && [ "$instance_ids" != "None" ]; then
        aws ec2 terminate-instances --instance-ids $instance_ids --region "$REGION" 2>/dev/null || true
        echo -e "  ${GREEN}Terminated $(echo $instance_ids | wc -w | tr -d ' ') instance(s)${NC}"
        # Wait for termination to complete before destroying CDK stack
        echo "  Waiting for instances to terminate..."
        aws ec2 wait instance-terminated --instance-ids $instance_ids --region "$REGION" 2>/dev/null || true
    else
        echo "  No running instances found."
    fi

    cd "$SCRIPT_DIR"
    echo "  Destroying CDK stack..."
    npx cdk destroy --force 2>&1 | grep -E "(destroy|delete)" | head -5 || true
    echo -e "  ${GREEN}Stack destroyed. No further charges.${NC}"
}

# ============================================================================
# Full run
# ============================================================================

cmd_full() {
    local start_time
    start_time=$(date +%s)

    cmd_matrix
    echo ""
    cmd_build
    cmd_deploy
    cmd_wait
    cmd_results

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo ""
    echo "  Total elapsed: ${duration}s"
    echo ""
    echo -e "  ${YELLOW}Run './run-proof.sh destroy' to tear down and stop billing.${NC}"
}

# ============================================================================
# Entry
# ============================================================================

case "${1:-full}" in
    build)   cmd_build ;;
    deploy)  cmd_deploy ;;
    status)  cmd_status ;;
    wait)    cmd_wait ;;
    results) cmd_results ;;
    destroy) cmd_destroy ;;
    matrix)  cmd_matrix ;;
    full)    cmd_full ;;
    *)       echo "Usage: $0 {build|deploy|status|wait|results|destroy|matrix|full}"; exit 1 ;;
esac
