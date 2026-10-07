#!/usr/bin/env bash
# Try to launch one on-demand GPU instance in ANY region/zone that has capacity.
# For each instance type (in order) and each enabled region that offers it:
#   - creates key pair "tau-forge-gpu" there if needed, saving the private key
#     to ~/.ssh/tau-forge-gpu-<region>.pem (never overwritten)
#   - creates security group "tau-forge-ssh" (SSH from your current IP only)
#   - picks the newest "Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023)"
#   - tries every subnet of the default VPC
# Stops at the first successful launch. Needs ec2:* on the IAM user.
#
# Usage: bash infra/launch_gpu_anywhere.sh
#        TYPES="g7e.2xlarge" bash infra/launch_gpu_anywhere.sh
set -uo pipefail

TYPES="${TYPES:-p5.4xlarge g7e.2xlarge}"
KEY=tau-forge-gpu
SG_NAME=tau-forge-ssh
NAME=tau-forge-gpu
MY_IP="$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')"
aws sts get-caller-identity >/dev/null || { echo "AWS CLI is not logged in."; exit 1; }
echo "Your IP: $MY_IP. Instance types, in order: $TYPES"

REGIONS=$(aws ec2 describe-regions --query 'Regions[].RegionName' --output text)

setup_region() {  # sets SG and PEM for $1, returns non-zero to skip the region
    local r="$1" vpc
    PEM="$HOME/.ssh/$KEY-$r.pem"
    if [ ! -s "$PEM" ]; then
        aws ec2 delete-key-pair --region "$r" --key-name "$KEY" >/dev/null 2>&1
        aws ec2 create-key-pair --region "$r" --key-name "$KEY" \
            --query KeyMaterial --output text > "$PEM.tmp" 2>/dev/null && [ -s "$PEM.tmp" ] ||
            { rm -f "$PEM.tmp"; echo "  can't create key pair in $r"; return 1; }
        mv "$PEM.tmp" "$PEM"; chmod 600 "$PEM"
    fi
    vpc=$(aws ec2 describe-vpcs --region "$r" --filters Name=is-default,Values=true \
        --query 'Vpcs[0].VpcId' --output text 2>/dev/null)
    [ -n "$vpc" ] && [ "$vpc" != "None" ] || { echo "  no default VPC in $r"; return 1; }
    VPC="$vpc"
    SG=$(aws ec2 describe-security-groups --region "$r" \
        --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$vpc" \
        --query 'SecurityGroups[0].GroupId' --output text)
    if [ "$SG" = "None" ] || [ -z "$SG" ]; then
        SG=$(aws ec2 create-security-group --region "$r" --group-name "$SG_NAME" \
            --description "SSH for tau-forge GPU box" --vpc-id "$vpc" --query GroupId --output text) ||
            { echo "  can't create security group in $r"; return 1; }
    fi
    aws ec2 authorize-security-group-ingress --region "$r" --group-id "$SG" \
        --protocol tcp --port 22 --cidr "$MY_IP/32" >/dev/null 2>&1 || true
}

for TYPE in $TYPES; do
    echo "######## $TYPE ########"
    for R in $REGIONS; do
        AZS=$(aws ec2 describe-instance-type-offerings --region "$R" --location-type availability-zone \
            --filters "Name=instance-type,Values=$TYPE" --query 'InstanceTypeOfferings[].Location' --output text 2>/dev/null)
        [ -n "$AZS" ] || continue
        echo "=== $R ($TYPE offered in: $AZS)"
        setup_region "$R" || continue
        AMI=$(aws ec2 describe-images --region "$R" --owners amazon \
            --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023)*" \
                      "Name=architecture,Values=x86_64" "Name=state,Values=available" \
            --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
        [ -n "$AMI" ] && [ "$AMI" != "None" ] || { echo "  no DLAMI in $R"; continue; }
        for AZ in $AZS; do
            SUBNET=$(aws ec2 describe-subnets --region "$R" \
                --filters "Name=vpc-id,Values=$VPC" "Name=availability-zone,Values=$AZ" \
                --query 'Subnets[0].SubnetId' --output text)
            [ -n "$SUBNET" ] && [ "$SUBNET" != "None" ] || continue
            OUT=$(aws ec2 run-instances --region "$R" --image-id "$AMI" --instance-type "$TYPE" \
                --key-name "$KEY" --security-group-ids "$SG" --subnet-id "$SUBNET" \
                --associate-public-ip-address \
                --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=200,VolumeType=gp3}' \
                --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
                --query 'Instances[0].InstanceId' --output text 2>&1)
            if [[ "$OUT" == i-* ]]; then
                echo "  LAUNCHED $TYPE $OUT in $AZ. Waiting for it to start..."
                aws ec2 wait instance-running --region "$R" --instance-ids "$OUT"
                IP=$(aws ec2 describe-instances --region "$R" --instance-ids "$OUT" \
                    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
                echo
                echo "SSH in (give it ~1 min to boot):"
                echo "  ssh -i $PEM ec2-user@$IP"
                echo "Stop it when idle:"
                echo "  aws ec2 stop-instances --region $R --instance-ids $OUT"
                exit 0
            fi
            CODE=$(echo "$OUT" | grep -o '([A-Za-z.]*)' | head -1)
            echo "  $AZ -> ${CODE:-$OUT}"
            case "$OUT" in
                *VcpuLimitExceeded*) echo "  (quota for this family is too low in $R; skipping region)"; break ;;
                *PendingVerification*) echo "  (AWS is still verifying your account for $R; usually minutes, up to 4 h. Rerun later.)"; break ;;
                *InsufficientInstanceCapacity*|*Unsupported*) ;;
                *) echo "  unexpected error: $OUT" ;;
            esac
        done
    done
done
echo "No capacity for [$TYPES] anywhere right now. Rerun later, e.g.:"
echo "  until bash $0; do sleep 300; done"
exit 1
