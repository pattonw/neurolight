from dataclasses import dataclass, field

from typing import List

@dataclass
class DataGen:
    shift_attempts: int = 1000
    request_attempts: int = 10
    seperate_by: List[int] = field(default_factory=lambda: [15000, 15500])
    num_components: int = 1
    guarantee_nonempty: bool = True
