from langchain_core.tools import tool
import uuid


@tool
def create_hardware_ticket(device_type: str, issue_summary: str):
    """
    Create a hardware support ticket for physical device-related issues.

    Args:
        device_type: Type of hardware device affected.
        issue_summary: Short description of the hardware issue.
    """

    ticket_id = f"HW-{str(uuid.uuid4())[:8].upper()}"

    return {
        "ticket_id": ticket_id,
        "team": "Hardware Support",
        "status": "Created",
        "request_summary": f"{device_type} issue reported: {issue_summary}"
    }


@tool
def create_access_request(system_name: str, access_type: str):
    """
    Create an access request ticket for systems/applications/resources.

    Args:
        system_name: Name of system user needs access to.
        access_type: Type of access required.
    """

    request_id = f"ACC-{str(uuid.uuid4())[:8].upper()}"

    return {
        "request_id": request_id,
        "team": "Access Management",
        "status": "Submitted",
        "request_summary": f"{access_type} requested for {system_name}"
    }


@tool
def request_software_installation(software_name: str):
    """
    Create a software installation request.

    Args:
        software_name: Name of software user wants installed.
    """

    request_id = f"SW-{str(uuid.uuid4())[:8].upper()}"

    return {
        "request_id": request_id,
        "team": "Software Support",
        "status": "Pending Approval",
        "request_summary": f"Software installation requested for {software_name}"
    }

ticket_creation_tools = [create_hardware_ticket,
                         create_access_request,
                         request_software_installation,
                        ]