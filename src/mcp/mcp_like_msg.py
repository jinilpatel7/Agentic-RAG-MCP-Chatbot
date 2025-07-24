# This file defines a standardized message structure for the Model Context Protocol (MCP).

from typing import Any, Dict

class MCPMessage:
    """
    A standardized message wrapper for communication between agents
    using the Model Context Protocol (MCP) pattern.
    
    This ensures that all messages share a common structure, which includes:
    - sender:     who sent the message
    - receiver:   who is the intended recipient
    - type:       what kind of message this is (e.g., RETRIEVAL_RESULT, LLM_RESPONSE)
    - trace_id:   unique ID to trace message flow through the system
    - payload:    the actual data being passed
    """

    def __init__(self, sender: str, receiver: str, msg_type: str, trace_id: str, payload: Dict[str, Any]):
        """
        Initializes an MCPMessage with consistent structure.

        Args:
            sender (str): The agent/component sending the message.
            receiver (str): The intended recipient agent/component.
            msg_type (str): The type or category of message.
            trace_id (str): Unique identifier for this request or message flow.
            payload (Dict[str, Any]): The data to be passed with this message.
        """
        self.message = {
            "sender": sender,
            "receiver": receiver,
            "type": msg_type,
            "trace_id": trace_id,
            "payload": payload
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the message object to a dictionary format for transport
        or logging.

        Output:
            Dict[str, Any]: The structured message dictionary.
        """
        return self.message
