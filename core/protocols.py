from dataclasses import dataclass


@dataclass
class SafetyProtocol:
    """
    Safety prompting configuration for MAP safety topology experiments.
    """
    system_rigid: str
    system_adaptive: str
    jailbreak_prompt: str
