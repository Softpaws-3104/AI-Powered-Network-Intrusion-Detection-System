
import argparse
from collections import Counter
from datetime import datetime, timezone

from scapy.all import sniff, IP, IPv6, TCP, UDP, ICMP, ARP


counts = Counter()
packet_limit = 100


def handle_packet(packet):
    """Record basic metadata only; do not store packet payloads."""
    counts["total"] += 1

    if IP in packet:
        counts["IPv4"] += 1
        counts[f"IPv4 protocol {packet[IP].proto}"] += 1
    elif IPv6 in packet:
        counts["IPv6"] += 1

    if TCP in packet:
        counts["TCP"] += 1
        flags = packet[TCP].flags
        for flag, name in [
            (0x01, "FIN"),
            (0x02, "SYN"),
            (0x04, "RST"),
            (0x08, "PSH"),
            (0x10, "ACK"),
            (0x40, "ECE"),
            (0x80, "CWR"),
        ]:
            if int(flags) & flag:
                counts[f"TCP {name}"] += 1

    if UDP in packet:
        counts["UDP"] += 1
    if ICMP in packet:
        counts["ICMP"] += 1
    if ARP in packet:
        counts["ARP"] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Short, bounded Scapy packet-capture test"
    )
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    if not 1 <= args.count <= 1000:
        parser.error("--count must be between 1 and 1000")

    print(f"Interface: {args.iface}")
    print(f"Maximum packets: {args.count}")
    print("Starting capture. Press Ctrl+C to stop early.")

    try:
        sniff(
            iface=args.iface,
            prn=handle_packet,
            store=False,
            count=args.count,
            timeout=30,
        )
    except PermissionError:
        print("Permission denied. Packet capture may require elevated privileges.")
        raise SystemExit(1)
    except OSError as exc:
        print(f"Capture failed: {exc}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nCapture stopped by user.")

    print("\nCapture summary")
    print("Finished at:", datetime.now(timezone.utc).isoformat())
    for name, value in sorted(counts.items()):
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
