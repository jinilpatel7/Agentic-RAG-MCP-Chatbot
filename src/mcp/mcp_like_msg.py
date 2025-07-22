from typing import Any, Dict

class MCPMessage:
    def __init__(self, sender: str, receiver: str, msg_type: str, trace_id: str, payload: Dict[str, Any]):
        self.message = {
            "sender": sender,
            "receiver": receiver,
            "type": msg_type,
            "trace_id": trace_id,
            "payload": payload
        }

    def to_dict(self):
        return self.message