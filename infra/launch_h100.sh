#!/usr/bin/env bash
# Launch one on-demand p5.4xlarge (1x H100 80GB), trying every availability
# zone until one has capacity. Creates nothing: it reuses the key pair and
# security group of an instance you already have (e.g. the A10G eval box),
# found automatically, so it works with an IAM user that can't create keys
# or security groups. Uses the newest "Deep Learning OSS Nvidia Driver AMI
# GPU PyTorch * (Amazon Linux 2023)".
#
# Usage: bash infra/launch_h100.sh [region]            (default: us-east-1)
# Other GPUs: TYPE=g7e.2xlarge bash infra/launch_h100.sh  (1x RTX PRO 6000 96GB, G-family quota)
# Override the auto-detection if needed:
#   KEY_NAME=my-key SG_ID=sg-0123 bash infra/launch_h100.sh us-east-2
set -uo pipefail

REGION="${1:-us-east-1}"
TYPE="${TYPE:-p5.4xlarge}"   # e.g. TYPE=g7e.2xlarge or TYPE=g6e.xlarge
NAME=tau-forge-h100

aws sts get-caller-identity >/dev/null || { echo "AWS CLI is not logged in."; exit 1; }

if [ -z "${KEY_NAME:-}" ] || [ -z "${SG_ID:-}" ]; then
    read -r REF_ID REF_KEY REF_SG REF_NAME <<<"$(aws ec2 describe-instances --region "$REGION" \
        --filters Name=instance-state-name,Values=running,stopped \
        --query 'sort_by(Reservations[].Instances[?KeyName!=`null`][], &LaunchTime)[-1].[InstanceId,KeyName,SecurityGroups[0].GroupId,Tags[?Key==`Name`]|[0].Value]' \
        --output text)"
    if [ -z "${REF_ID:-}" ] || [ "$REF_ID" = "None" ]; then
        echo "No existing instance with a key pair in $REGION to copy settings from."
        echo "Set them yourself: KEY_NAME=<key pair name> SG_ID=<sg-...> bash $0 $REGION"
        exit 1
    fi
    KEY_NAME="${KEY_NAME:-$REF_KEY}"
    SG_ID="${SG_ID:-$REF_SG}"
    echo "Copying settings from $REF_ID ($REF_NAME): key pair '$KEY_NAME', security group $SG_ID"
fi

AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023)*" \
              "Name=architecture,Values=x86_64" "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].[ImageId,Name]' --output text)
read -r AMI_ID AMI_NAME <<<"$AMI"
if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then echo "No DLAMI found in $REGION."; exit 1; fi
echo "AMI: $AMI_ID ($AMI_NAME)"

VPC=$(aws ec2 describe-security-groups --region "$REGION" --group-ids "$SG_ID" \
    --query 'SecurityGroups[0].VpcId' --output text)

for SUBNET in $(aws ec2 describe-subnets --region "$REGION" --filters "Name=vpc-id,Values=$VPC" \
        --query 'Subnets[].SubnetId' --output text); do
    echo "Trying $SUBNET ..."
    OUT=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI_ID" --instance-type "$TYPE" \
        --key-name "$KEY_NAME" --security-group-ids "$SG_ID" --subnet-id "$SUBNET" \
        --associate-public-ip-address \
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=200,VolumeType=gp3}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
        --query 'Instances[0].InstanceId' --output text 2>&1)
    if [[ "$OUT" == i-* ]]; then
        echo "Launched $OUT. Waiting for it to start..."
        aws ec2 wait instance-running --region "$REGION" --instance-ids "$OUT"
        IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$OUT" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
        echo
        echo "SSH in (give it ~1 min to boot), with the same .pem you use for key pair '$KEY_NAME':"
        echo "  ssh -i <path to $KEY_NAME .pem> ec2-user@$IP"
        echo "Stop it when idle:"
        echo "  aws ec2 stop-instances --region $REGION --instance-ids $OUT"
        exit 0
    fi
    CODE=$(echo "$OUT" | grep -o 'An error occurred ([A-Za-z.]*)' | head -1)
    echo "  -> ${CODE:-$OUT}"
    case "$OUT" in
        *InsufficientInstanceCapacity*|*Unsupported*|*InvalidSubnet*) continue ;;
        *VcpuLimitExceeded*)
            echo "Quota too low in $REGION: P instances need the "Running On-Demand P instances" quota, G instances "Running On-Demand G and VT instances"."; exit 1 ;;
        *) echo "Not a capacity problem, stopping. Full error:"; echo "$OUT"; exit 1 ;;
    esac
done
echo "No $TYPE capacity in any $REGION zone right now. Retry in a while, or try another region"
echo "(key pairs and security groups are per region, so pass KEY_NAME/SG_ID that exist there)."
exit 1
