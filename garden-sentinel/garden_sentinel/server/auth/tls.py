"""
TLS/SSL configuration for Garden Sentinel.

Provides:
- Self-signed certificate generation for development
- Certificate loading for production
- Secure SSL context configuration
"""

import os
import ssl
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TLSConfig:
    """TLS configuration."""
    enabled: bool = True
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None  # For client certificate validation

    # Auto-generate self-signed cert if files don't exist
    auto_generate: bool = True
    cert_dir: str = "/etc/garden-sentinel/certs"

    # Certificate details for auto-generation
    common_name: str = "garden-sentinel"
    organization: str = "Garden Sentinel"
    validity_days: int = 365

    # SSL options
    minimum_version: str = "TLSv1.2"
    verify_client: bool = False  # Require client certificates


def generate_self_signed_cert(
    cert_file: str,
    key_file: str,
    common_name: str = "garden-sentinel",
    organization: str = "Garden Sentinel",
    validity_days: int = 365,
    san_dns: Optional[list[str]] = None,
    san_ip: Optional[list[str]] = None,
) -> bool:
    """
    Generate a self-signed certificate using OpenSSL.

    Returns True if successful.
    """
    cert_path = Path(cert_file)
    key_path = Path(key_file)

    # Create directory if needed
    cert_path.parent.mkdir(parents=True, exist_ok=True)

    # Build subject alternative names
    san_entries = []
    if san_dns:
        for i, dns in enumerate(san_dns):
            san_entries.append(f"DNS.{i+1}={dns}")
    else:
        san_entries.append(f"DNS.1={common_name}")
        san_entries.append("DNS.2=localhost")

    if san_ip:
        for i, ip in enumerate(san_ip):
            san_entries.append(f"IP.{i+1}={ip}")
    else:
        san_entries.append("IP.1=127.0.0.1")

    san_string = ",".join(san_entries)

    # Generate using OpenSSL
    try:
        # Create OpenSSL config for SAN
        config_content = f"""
[req]
default_bits = 4096
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = {organization}
CN = {common_name}

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = {san_string}
"""

        config_path = cert_path.parent / "openssl.cnf"
        with open(config_path, "w") as f:
            f.write(config_content)

        # Generate certificate and key
        result = subprocess.run(
            [
                "openssl", "req", "-x509", "-nodes",
                "-days", str(validity_days),
                "-newkey", "rsa:4096",
                "-keyout", str(key_path),
                "-out", str(cert_path),
                "-config", str(config_path),
            ],
            capture_output=True,
            text=True,
        )

        # Clean up config
        config_path.unlink(missing_ok=True)

        if result.returncode != 0:
            print(f"OpenSSL error: {result.stderr}")
            return False

        # Set permissions
        os.chmod(key_path, 0o600)
        os.chmod(cert_path, 0o644)

        return True

    except FileNotFoundError:
        print("OpenSSL not found. Please install OpenSSL.")
        return False
    except Exception as e:
        print(f"Error generating certificate: {e}")
        return False


def create_ssl_context(config: TLSConfig) -> Optional[ssl.SSLContext]:
    """
    Create an SSL context from configuration.

    Returns None if TLS is disabled or configuration is invalid.
    """
    if not config.enabled:
        return None

    cert_file = config.cert_file or f"{config.cert_dir}/server.crt"
    key_file = config.key_file or f"{config.cert_dir}/server.key"

    # Auto-generate if needed
    if config.auto_generate:
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            print(f"Generating self-signed certificate...")
            success = generate_self_signed_cert(
                cert_file=cert_file,
                key_file=key_file,
                common_name=config.common_name,
                organization=config.organization,
                validity_days=config.validity_days,
            )
            if not success:
                print("Failed to generate certificate. Running without TLS.")
                return None

    # Verify files exist
    if not os.path.exists(cert_file):
        print(f"Certificate file not found: {cert_file}")
        return None
    if not os.path.exists(key_file):
        print(f"Key file not found: {key_file}")
        return None

    # Create context
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # Set minimum version
        if config.minimum_version == "TLSv1.2":
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        elif config.minimum_version == "TLSv1.3":
            context.minimum_version = ssl.TLSVersion.TLSv1_3

        # Load certificate
        context.load_cert_chain(cert_file, key_file)

        # Load CA for client verification
        if config.verify_client and config.ca_file:
            context.load_verify_locations(config.ca_file)
            context.verify_mode = ssl.CERT_REQUIRED
        else:
            context.verify_mode = ssl.CERT_NONE

        # Security settings
        context.set_ciphers("ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20")
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1

        return context

    except Exception as e:
        print(f"Error creating SSL context: {e}")
        return None


def create_client_ssl_context(
    verify: bool = True,
    ca_file: Optional[str] = None,
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None,
) -> ssl.SSLContext:
    """
    Create an SSL context for client connections (edge -> server).

    Args:
        verify: Whether to verify server certificate
        ca_file: CA certificate file for verification
        cert_file: Client certificate for mutual TLS
        key_file: Client key for mutual TLS
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    if verify:
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True

        if ca_file:
            context.load_verify_locations(ca_file)
        else:
            context.load_default_certs()
    else:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    # Load client certificate for mutual TLS
    if cert_file and key_file:
        context.load_cert_chain(cert_file, key_file)

    # Security settings
    context.minimum_version = ssl.TLSVersion.TLSv1_2

    return context


def get_cert_info(cert_file: str) -> dict:
    """Get information about a certificate."""
    try:
        result = subprocess.run(
            ["openssl", "x509", "-in", cert_file, "-noout", "-text"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return {"error": result.stderr}

        # Parse output
        output = result.stdout
        info = {
            "raw": output,
        }

        # Extract key fields
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("Subject:"):
                info["subject"] = line.replace("Subject:", "").strip()
            elif line.startswith("Issuer:"):
                info["issuer"] = line.replace("Issuer:", "").strip()
            elif line.startswith("Not Before:"):
                info["not_before"] = line.replace("Not Before:", "").strip()
            elif line.startswith("Not After:"):
                info["not_after"] = line.replace("Not After:", "").strip()

        return info

    except Exception as e:
        return {"error": str(e)}
