"""
SageMaker Async Endpoint Deployment Script for Qwen3-Omni-30B-A3B
Run this from a SageMaker notebook instance.

Usage:
    import sagemaker
    from deploy_async import deploy_async_endpoint, invoke_async, poll_result, cleanup_endpoint

    IMAGE_URI = "123456789.dkr.ecr.us-east-1.amazonaws.com/qwen-3-omni-infer:latest"
    ROLE = sagemaker.get_execution_role()
    ASYNC_OUTPUT_S3 = "s3://my-bucket/async-inference/output/"

    # Deploy base model
    deploy_async_endpoint(
        endpoint_name="qwen3-omni-base",
        image_uri=IMAGE_URI,
        role=ROLE,
        async_output_s3=ASYNC_OUTPUT_S3,
    )

    # Deploy with LoRA adapter
    deploy_async_endpoint(
        endpoint_name="qwen3-omni-ft-v1",
        image_uri=IMAGE_URI,
        role=ROLE,
        async_output_s3=ASYNC_OUTPUT_S3,
        adapter_s3_uri="s3://my-bucket/adapters/v1/",
    )

    # Invoke
    result = invoke_async("qwen3-omni-ft-v1", payload, async_output_s3=ASYNC_OUTPUT_S3)
    output = poll_result(result["output_location"])

    # Cleanup
    cleanup_endpoint("qwen3-omni-ft-v1")
"""

import json
import time
import uuid
import boto3


# =============================================================================
# Deployment
# =============================================================================

def deploy_async_endpoint(
    endpoint_name: str,
    image_uri: str,
    role: str,
    async_output_s3: str,
    async_error_s3: str = None,
    adapter_s3_uri: str = None,
    instance_type: str = "ml.g6.12xlarge",
    initial_instance_count: int = 1,
    volume_size_gb: int = 200,
    model_loading_timeout: int = 900,
    sns_success_topic: str = None,
    sns_error_topic: str = None,
    enable_autoscaling: bool = True,
    autoscale_min_instances: int = 0,
    autoscale_max_instances: int = 10,
    autoscale_target_backlog_per_instance: int = 5,
    autoscale_scale_in_cooldown: int = 300,
    autoscale_scale_out_cooldown: int = 60,
    wait: bool = True,
) -> str:
    """Deploy an async SageMaker endpoint for Qwen3-Omni.

    Args:
        endpoint_name: Unique name for the endpoint.
        image_uri: ECR image URI for the inference container.
        role: IAM role ARN for SageMaker (needs S3, ECR, Secrets Manager access).
        async_output_s3: S3 path for async inference output.
        async_error_s3: S3 path for async inference errors. Defaults to async_output_s3 sibling.
        adapter_s3_uri: S3 URI to LoRA adapter directory. If None, deploys base model.
        instance_type: EC2 instance type.
        initial_instance_count: Starting number of instances.
        volume_size_gb: EBS volume size for model weights + tmp files.
        model_loading_timeout: Seconds allowed for model loading / container startup.
        sns_success_topic: SNS topic ARN for successful inference notifications.
        sns_error_topic: SNS topic ARN for failed inference notifications.
        enable_autoscaling: Whether to attach autoscaling policies.
        autoscale_min_instances: Minimum instance count (0 for scale-to-zero).
        autoscale_max_instances: Maximum instance count.
        autoscale_target_backlog_per_instance: Target backlog size per instance for scaling.
        autoscale_scale_in_cooldown: Seconds before allowing another scale-in action.
        autoscale_scale_out_cooldown: Seconds before allowing another scale-out action.
        wait: Whether to block until endpoint is InService.

    Returns:
        The endpoint name.
    """
    if async_error_s3 is None:
        async_error_s3 = async_output_s3.rstrip("/").rsplit("/", 1)[0] + "/error/"

    sm_client = boto3.client("sagemaker")

    print(f"{'='*60}")
    print(f"Deploying endpoint: {endpoint_name}")
    print(f"Instance type:      {instance_type}")
    print(f"Adapter:            {adapter_s3_uri or 'None (base model)'}")
    print(f"Autoscaling:        {autoscale_min_instances}-{autoscale_max_instances} instances")
    print(f"{'='*60}")

    # --- Environment variables ---
    env = {
        "HF_HOME": "/tmp/hf_cache",
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
    }
    if adapter_s3_uri:
        env["ADAPTER_S3_URI"] = adapter_s3_uri

    # --- Model ---
    model_name = endpoint_name

    _delete_if_exists(sm_client, model_name=model_name)

    print("Creating model...")
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "Environment": env,
        },
        ExecutionRoleArn=role,
    )

    # --- Endpoint Config ---
    config_name = endpoint_name

    print("Creating endpoint config...")
    async_config = {
        "OutputConfig": {
            "S3OutputPath": async_output_s3,
            "S3FailurePath": async_error_s3,
        },
        # Set to 1 — video inference is GPU-memory-heavy.
        "MaxConcurrentInvocationsPerInstance": 1,
    }
    if sns_success_topic or sns_error_topic:
        notification_config = {}
        if sns_success_topic:
            notification_config["SuccessTopic"] = sns_success_topic
        if sns_error_topic:
            notification_config["ErrorTopic"] = sns_error_topic
        async_config["OutputConfig"]["NotificationConfig"] = notification_config

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": initial_instance_count,
                "VolumeSizeInGB": volume_size_gb,
                "ModelDataDownloadTimeoutInSeconds": model_loading_timeout,
                "ContainerStartupHealthCheckTimeoutInSeconds": model_loading_timeout,
            }
        ],
        AsyncInferenceConfig=async_config,
    )

    # --- Endpoint ---
    print("Creating endpoint (this will take 10-15 min for initial deployment)...")
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )

    if wait:
        _wait_for_endpoint(sm_client, endpoint_name)

    # --- Autoscaling ---
    if enable_autoscaling:
        _setup_autoscaling(
            endpoint_name=endpoint_name,
            min_instances=autoscale_min_instances,
            max_instances=autoscale_max_instances,
            target_backlog_per_instance=autoscale_target_backlog_per_instance,
            scale_in_cooldown=autoscale_scale_in_cooldown,
            scale_out_cooldown=autoscale_scale_out_cooldown,
        )

    print(f"\nEndpoint ready: {endpoint_name}")
    return endpoint_name


# =============================================================================
# Autoscaling
# =============================================================================

def _setup_autoscaling(
    endpoint_name: str,
    min_instances: int,
    max_instances: int,
    target_backlog_per_instance: int,
    scale_in_cooldown: int,
    scale_out_cooldown: int,
):
    """Configure autoscaling on the async endpoint.

    Three-policy pattern required for proper scale-to/from-zero behavior:
      1. scale-from-zero: Step scaling triggered by HasBacklogWithoutCapacity CW alarm
      2. scale-on-backlog: Target tracking on ApproximateBacklogSizePerInstance (1→N),
         with scale-in disabled (we handle scale-to-zero explicitly)
      3. scale-to-zero: Step scaling triggered when backlog is empty for 10 min
    """
    aas_client = boto3.client("application-autoscaling")
    cw_client = boto3.client("cloudwatch")

    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    print(f"\nConfiguring autoscaling: {min_instances}-{max_instances} instances...")

    # Clean up any existing policies (safe for redeploys)
    for policy_name in [
        f"{endpoint_name}-scale-from-zero",
        f"{endpoint_name}-scale-on-backlog",
        f"{endpoint_name}-scale-to-zero",
    ]:
        try:
            aas_client.delete_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            )
        except Exception:
            pass

    # Register the scalable target
    aas_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=min_instances,
        MaxCapacity=max_instances,
    )

    # ---- Policy 1: Scale from zero ----
    response = aas_client.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scale-from-zero",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ExactCapacity",
            "StepAdjustments": [
                {
                    "MetricIntervalLowerBound": 0,
                    "ScalingAdjustment": 1,
                }
            ],
            "Cooldown": scale_out_cooldown,
        },
    )
    scale_from_zero_arn = response["PolicyARN"]

    cw_client.put_metric_alarm(
        AlarmName=f"{endpoint_name}-scale-from-zero-alarm",
        MetricName="HasBacklogWithoutCapacity",
        Namespace="AWS/SageMaker",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        Statistic="Average",
        Period=60,
        EvaluationPeriods=1,
        Threshold=0.5,
        ComparisonOperator="GreaterThanThreshold",
        AlarmActions=[scale_from_zero_arn],
        TreatMissingData="notBreaching",
    )

    # ---- Policy 2: Scale on backlog (1 → N) ----
    aas_client.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scale-on-backlog",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": float(target_backlog_per_instance),
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [
                    {"Name": "EndpointName", "Value": endpoint_name},
                ],
                "Statistic": "Average",
            },
            "ScaleInCooldown": scale_in_cooldown,
            "ScaleOutCooldown": scale_out_cooldown,
            "DisableScaleIn": True,
        },
    )

    # ---- Policy 3: Scale to zero ----
    response = aas_client.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scale-to-zero",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ExactCapacity",
            "StepAdjustments": [
                {
                    "MetricIntervalUpperBound": 0,
                    "ScalingAdjustment": 0,
                }
            ],
            "Cooldown": scale_in_cooldown,
        },
    )
    scale_to_zero_arn = response["PolicyARN"]

    cw_client.put_metric_alarm(
        AlarmName=f"{endpoint_name}-scale-to-zero-alarm",
        MetricName="ApproximateBacklogSize",
        Namespace="AWS/SageMaker",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        Statistic="Average",
        Period=300,
        EvaluationPeriods=2,
        Threshold=0.5,
        ComparisonOperator="LessThanThreshold",
        AlarmActions=[scale_to_zero_arn],
        TreatMissingData="notBreaching",
    )

    print("Autoscaling configured:")
    print(f"  scale-from-zero:  HasBacklogWithoutCapacity > 0.5 → set to 1 instance")
    print(f"  scale-on-backlog: target {target_backlog_per_instance} requests/instance (1→N, scale-in disabled)")
    print(f"  scale-to-zero:    backlog < 0.5 for 10 min → set to 0 instances")


# =============================================================================
# Invocation helpers
# =============================================================================

def invoke_async(endpoint_name: str, payload: dict, async_output_s3: str) -> dict:
    """Send an async inference request.

    Args:
        endpoint_name: Name of the deployed endpoint.
        payload: Request payload dict (s3_uri, user_prompt, etc.)
        async_output_s3: S3 path used for async output (needed to derive input upload bucket).

    Returns:
        Dict with output_location and inference_id.
    """
    sm_runtime = boto3.client("sagemaker-runtime")

    input_location = _upload_payload_to_s3(payload, async_output_s3)

    response = sm_runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        ContentType="application/json",
        InputLocation=input_location,
    )

    output_location = response["OutputLocation"]
    print(f"Async request submitted. Output will be at: {output_location}")
    return {
        "output_location": output_location,
        "inference_id": response.get("InferenceId"),
    }


def _upload_payload_to_s3(payload: dict, async_output_s3: str) -> str:
    """Upload request payload to S3 for async invocation."""
    s3 = boto3.client("s3")

    output_path = async_output_s3[5:]  # strip s3://
    bucket = output_path.split("/", 1)[0]

    key = f"async-inference/input/{uuid.uuid4()}.json"

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload),
        ContentType="application/json",
    )

    return f"s3://{bucket}/{key}"


def poll_result(output_location: str, poll_interval: int = 10, timeout: int = 600) -> dict:
    """Poll S3 until async result is available.

    Args:
        output_location: S3 URI from invoke_async response.
        poll_interval: Seconds between polls.
        timeout: Max seconds to wait.

    Returns:
        Parsed JSON result.
    """
    s3 = boto3.client("s3")

    path = output_location[5:]
    bucket = path.split("/", 1)[0]
    key = path.split("/", 1)[1]

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read().decode("utf-8")
            return json.loads(body)
        except s3.exceptions.NoSuchKey:
            print(f"  Waiting for result... ({int(time.time() - start)}s elapsed)")
            time.sleep(poll_interval)

    raise TimeoutError(f"Result not available after {timeout}s at {output_location}")


# =============================================================================
# Cleanup
# =============================================================================

def cleanup_endpoint(endpoint_name: str):
    """Delete endpoint, endpoint config, model, autoscaling policies, and CW alarms."""
    sm_client = boto3.client("sagemaker")
    aas_client = boto3.client("application-autoscaling")
    cw_client = boto3.client("cloudwatch")

    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    # Delete scaling policies
    for policy_name in [
        f"{endpoint_name}-scale-from-zero",
        f"{endpoint_name}-scale-on-backlog",
        f"{endpoint_name}-scale-to-zero",
    ]:
        try:
            aas_client.delete_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            )
            print(f"Deleted scaling policy: {policy_name}")
        except Exception:
            pass

    # Deregister autoscaling target
    try:
        aas_client.deregister_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        )
        print(f"Deregistered autoscaling for {endpoint_name}")
    except Exception:
        pass

    # Delete CloudWatch alarms
    for alarm_name in [
        f"{endpoint_name}-scale-from-zero-alarm",
        f"{endpoint_name}-scale-to-zero-alarm",
    ]:
        try:
            cw_client.delete_alarms(AlarmNames=[alarm_name])
            print(f"Deleted CW alarm: {alarm_name}")
        except Exception:
            pass

    # Delete endpoint
    try:
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Deleting endpoint: {endpoint_name}")
    except Exception as e:
        print(f"Could not delete endpoint: {e}")

    # Delete endpoint config
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Deleted endpoint config: {endpoint_name}")
    except Exception as e:
        print(f"Could not delete endpoint config: {e}")

    # Delete model
    try:
        sm_client.delete_model(ModelName=endpoint_name)
        print(f"Deleted model: {endpoint_name}")
    except Exception as e:
        print(f"Could not delete model: {e}")


# =============================================================================
# Internal helpers
# =============================================================================

def _delete_if_exists(sm_client, model_name: str):
    """Delete existing model/config/endpoint if they exist (for redeployment)."""
    try:
        sm_client.describe_endpoint(EndpointName=model_name)
        print(f"Endpoint {model_name} exists, deleting...")
        sm_client.delete_endpoint(EndpointName=model_name)
        waiter = sm_client.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=model_name, WaiterConfig={"Delay": 15, "MaxAttempts": 60})
    except sm_client.exceptions.ClientError:
        pass

    try:
        sm_client.delete_endpoint_config(EndpointConfigName=model_name)
    except sm_client.exceptions.ClientError:
        pass

    try:
        sm_client.delete_model(ModelName=model_name)
    except sm_client.exceptions.ClientError:
        pass


def _wait_for_endpoint(sm_client, endpoint_name: str, poll_interval: int = 30):
    """Wait for endpoint to be InService."""
    while True:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]

        if status == "InService":
            print(f"\nEndpoint is InService!")
            return
        elif status == "Failed":
            reason = resp.get("FailureReason", "Unknown")
            raise RuntimeError(f"Endpoint creation failed: {reason}")
        else:
            print(f"  Status: {status}...")
            time.sleep(poll_interval)
