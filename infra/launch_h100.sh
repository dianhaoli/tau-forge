#!/usr/bin/env bash
# Launch one on-demand p5.4xlarge (1x H100 80GB), trying every availability
# zone in each region until one has capacity. Sets up everything it needs:
#   - an SSH key (~/.ssh/tau-forge-h100), imported into each region it tries
#   - a security group "tau-forge-ssh" allowing SSH from your current IP only
#   - the newest "Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023)"
# Needs only the AWS CLI, logged in (`aws sts get-caller-identity` works).
#
# Usage: ./infra/launch_h100.sh [region ...]   (default: us-east-1 us-east-2 us-west-2)
set -uo pipefail

REGIONS=("$@")
[ $# -eq 0 ] && REGIONS=(us-east-1 us-east-2 us-west-2)
TYPE=p5.4xlarge
NAME=tau-forge-h100
KEY_FILE="$HOME/.ssh/$NAME"
SG_NAME=tau-forge-ssh

aws sts get-caller-identity >/dev/null || { echo "AWS CLI is not logged in."; exit 1; }

if [ ! -f "$KEY_FILE" ]; then
    mkdir -p "$HOME/.ssh"
    ssh-keygen -t ed25519 -N "" -f "$KEY_FILE" -C "$NAME" >/dev/null
    echo "Created SSH key $KEY_FILE"
fi
MY_IP="$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')"
echo "Your IP: $MY_IP"

for REGION in "${REGIONS[@]}"; do
    echo "=== $REGION ==="
    AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon \
        --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023)*" \
                  "Name=architecture,Values=x86_64" "Name=state,Values=available" \
        --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
    if [ -z "$AMI" ] || [ "$AMI" = "None" ]; then echo "No DLAMI found in $REGION, skipping."; continue; fi
    echo "AMI: $AMI"

    aws ec2 describe-key-pairs --region "$REGION" --key-names "$NAME" >/dev/null 2>&1 ||
        aws ec2 import-key-pair --region "$REGION" --key-name "$NAME" \
            --public-key-material "fileb://$KEY_FILE.pub" >/dev/null

    VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=is-default,Values=true \
        --query 'Vpcs[0].VpcId' --output text)
    if [ "$VPC" = "None" ]; then echo "No default VPC in $REGION, skipping."; continue; fi
    SG=$(aws ec2 describe-security-groups --region "$REGION" \
        --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC" \
        --query 'SecurityGroups[0].GroupId' --output text)
    if [ "$SG" = "None" ]; then
        SG=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" \
            --description "SSH for tau-forge GPU box" --vpc-id "$VPC" --query GroupId --output text)
    fi
    aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
        --protocol tcp --port 22 --cidr "$MY_IP/32" >/dev/null 2>&1 || true

    for SUBNET in $(aws ec2 describe-subnets --region "$REGION" \
            --filters "Name=vpc-id,Values=$VPC" Name=default-for-az,Values=true \
            --query 'Subnets[].SubnetId' --output text); do
        echo "Trying $SUBNET ..."
        OUT=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" \
            --key-name "$NAME" --security-group-ids "$SG" --subnet-id "$SUBNET" \
            --associate-public-ip-address \
            --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=200,VolumeType=gp3}' \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
            --query 'Instances[0].InstanceId' --output text 2>&1)
        if [[ "$OUT" == i-* ]]; then
            echo "Launched $OUT in $REGION. Waiting for it to start..."
            aws ec2 wait instance-running --region "$REGION" --instance-ids "$OUT"
            IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$OUT" \
                --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
            echo
            echo "SSH in (give it ~1 min to boot):"
            echo "  ssh -i $KEY_FILE ec2-user@$IP"
            echo "Stop it when idle:"
            echo "  aws ec2 stop-instances --region $REGION --instance-ids $OUT"
            exit 0
        fi
        echo "  -> $(echo "$OUT" | grep -o 'An error occurred ([A-Za-z]*)' || echo "$OUT" | tail -1)"
        if echo "$OUT" | grep -q "VcpuLimitExceeded"; then
            echo "  P-instance quota too low in $REGION (need 16 vCPUs of 'Running On-Demand P instances')."
            break
        fi
    done
done
echo "No capacity anywhere tried. Retry later, or try other regions: ./infra/launch_h100.sh us-west-1 eu-west-2"
exit 1
