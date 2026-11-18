"""The few RunPod GraphQL calls the fleet needs, standard library only."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

ENDPOINT = "https://api.runpod.io/graphql"
USER_AGENT = "deepstarnoodles-backcast/0.1"


class SupplyError(RuntimeError):
    """RunPod had no capacity for this request; nothing was created."""


def api_key() -> str:
    """RUNPOD_API_KEY, or the key `runpodctl config` saved in ~/.runpod/config.toml."""
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    config = Path.home() / ".runpod" / "config.toml"
    if config.exists():
        match = re.search(r'apikey\s*=\s*"([^"]+)"', config.read_text())
        if match:
            return match.group(1)
    raise RuntimeError("No RunPod API key: export RUNPOD_API_KEY or run `runpodctl config --apiKey ...`")


def utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


class RunPod:
    def __init__(self, key: str | None = None) -> None:
        self.key = key or api_key()

    def gql(self, query: str, variables: dict | None = None) -> dict:
        request = urllib.request.Request(
            ENDPOINT, data=json.dumps({"query": query, "variables": variables or {}}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}", "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            if "SUPPLY_CONSTRAINT" in detail or "no longer any instances" in detail:
                raise SupplyError(detail[:300]) from None
            raise RuntimeError(f"RunPod API HTTP {exc.code}: {detail[:300]}") from None
        if body.get("errors"):
            text = json.dumps(body["errors"])
            if "SUPPLY_CONSTRAINT" in text or "no longer any instances" in text:
                raise SupplyError(text[:300])
            raise RuntimeError(f"RunPod API error: {text[:500]}")
        return body["data"]

    def account(self) -> dict:
        return self.gql("query { myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr } } }")["myself"]

    def quote(self, gpu: str, cloud: str, min_vcpu: int, min_memory_gb: int) -> float | None:
        """Current lowest on-demand price, or None when nothing deployable is in stock."""
        query = """query($id:String,$sec:Boolean,$v:Int!,$m:Int!){gpuTypes(input:{id:$id}){id
          lowestPrice(input:{gpuCount:1,secureCloud:$sec,minMemoryInGb:$m,minVcpuCount:$v,minCudaVersion:"12.8"}){uninterruptablePrice stockStatus}}}"""
        rows = self.gql(query, {"id": gpu, "sec": cloud == "SECURE", "v": min_vcpu, "m": min_memory_gb})["gpuTypes"]
        return ((rows[0].get("lowestPrice") or {}).get("uninterruptablePrice") if rows else None)

    def deploy(self, *, name: str, gpu: str, cloud: str, image: str, container_gb: int, min_vcpu: int, min_memory_gb: int,
               public_key: str, stop_epoch: float, terminate_epoch: float) -> dict:
        query = """mutation($in:PodFindAndDeployOnDemandInput!){podFindAndDeployOnDemand(input:$in){
          id name costPerHr desiredStatus machine{gpuDisplayName dataCenterId}}}"""
        pod = self.gql(query, {"in": {
            "cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu, "name": name, "imageName": image, "dockerArgs": "",
            "containerDiskInGb": container_gb, "volumeInGb": 0, "volumeMountPath": "/workspace",
            "minVcpuCount": min_vcpu, "minMemoryInGb": min_memory_gb,
            "minCudaVersion": "12.8", "ports": "22/tcp", "supportPublicIp": True, "startSsh": True,
            "stopAfter": utc(stop_epoch), "terminateAfter": utc(terminate_epoch),
            "env": [{"key": "PUBLIC_KEY", "value": public_key}]}})["podFindAndDeployOnDemand"]
        if not pod or pod.get("name") != name or not pod.get("id"):
            raise RuntimeError("Ambiguous pod creation response")
        return pod

    def pod(self, pod_id: str) -> dict | None:
        query = """query($id:String!){pod(input:{podId:$id}){id name desiredStatus costPerHr
          runtime{uptimeInSeconds ports{ip isIpPublic privatePort publicPort type}}}}"""
        return self.gql(query, {"id": pod_id})["pod"]

    def terminate(self, pod_id: str) -> None:
        self.gql("mutation($id:String!){podTerminate(input:{podId:$id})}", {"id": pod_id})


def ssh_endpoint(pod: dict | None) -> tuple[str, int] | None:
    for port in ((pod or {}).get("runtime") or {}).get("ports") or []:
        if port["privatePort"] == 22 and port["isIpPublic"]:
            return port["ip"], port["publicPort"]
    return None
