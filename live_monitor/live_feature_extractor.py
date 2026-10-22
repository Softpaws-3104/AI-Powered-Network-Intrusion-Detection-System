"""Bounded live packet capture and 10-packet feature aggregation."""

from collections import Counter
from statistics import mode
import time

import numpy as np
import pandas as pd
from scapy.all import sniff, Ether, IP, TCP, UDP, ARP, ICMP, IPv6, LLC

FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count",
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP",
    "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP",
    "IGMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG",
    "Std", "Tot size", "IAT", "Number", "Variance",
]

BATCH_SIZE = 10


def packet_features(packet, previous_timestamp=None):
    """Convert one captured packet into the original per-packet schema."""
    ts = float(getattr(packet, "time", time.time()))
    size = len(packet)

    row = {name: 0.0 for name in FEATURES}
    row["Tot size"] = float(size)
    row["Number"] = 1.0
    row["IAT"] = (
        max(0.0, ts - previous_timestamp)
        if previous_timestamp is not None else 0.0
    )

    if IP in packet:
        ip = packet[IP]
        row["Protocol Type"] = float(ip.proto)
        row["Time_To_Live"] = float(ip.ttl)
        row["IPv"] = 1.0
        if ip.proto == 1:
            row["ICMP"] = 1.0
        elif ip.proto == 2:
            row["IGMP"] = 1.0
    if IPv6 in packet:
        # The original extractor discards IPv6 packets.
        return None, ts


    row["LLC"] = 1.0

    if ARP in packet:
        row["ARP"] = 1.0

    if ICMP in packet:
        row["ICMP"] = 1.0

    if TCP in packet:
        tcp = packet[TCP]
        row["TCP"] = 1.0
        row["Header_Length"] = float((tcp.dataofs or 5) * 4)
        flags = int(tcp.flags)

        flag_map = {
            "fin_flag_number": 0x01,
            "syn_flag_number": 0x02,
            "rst_flag_number": 0x04,
            "psh_flag_number": 0x08,
            "ack_flag_number": 0x10,
            "ece_flag_number": 0x40,
            "cwr_flag_number": 0x80,
        }
        for name, bit in flag_map.items():
            row[name] = float(bool(flags & bit))

        row["ack_count"] = row["ack_flag_number"]
        row["syn_count"] = row["syn_flag_number"]
        row["fin_count"] = row["fin_flag_number"]
        row["rst_count"] = row["rst_flag_number"]

    if UDP in packet:
        udp = packet[UDP]
        row["UDP"] = 1.0
        row["Header_Length"] = 8.0

    # Infer application protocols from standard ports, as in a
    # lightweight live approximation of the original extractor.
    ports = set()
    if TCP in packet:
        ports.update([int(packet[TCP].sport), int(packet[TCP].dport)])
    if UDP in packet:
        ports.update([int(packet[UDP].sport), int(packet[UDP].dport)])

    row["HTTP"] = float(80 in ports)
    row["HTTPS"] = float(443 in ports)
    row["DNS"] = float(53 in ports)
    row["Telnet"] = float(23 in ports)
    row["SMTP"] = float(25 in ports)
    row["SSH"] = float(22 in ports)
    row["IRC"] = float(21 in ports)

    if UDP in packet:
        udp = packet[UDP]
        row["DHCP"] = float(
            (int(udp.sport) == 67 and int(udp.dport) == 68)
            or (int(udp.sport) == 68 and int(udp.dport) == 67)
        )

    # Keep packet-size statistics initialized at the per-packet stage;
    # the 10-packet aggregation below calculates them.
    return row, ts


def aggregate_batch(rows):
    """Aggregate a complete batch using the original 10-row rules."""
    if len(rows) != BATCH_SIZE:
        raise ValueError(f"Expected {BATCH_SIZE} packets, received {len(rows)}")

    timestamps = [float(row["_timestamp"]) for row in rows]
    df = pd.DataFrame(
        [{name: row[name] for name in FEATURES} for row in rows],
        columns=FEATURES,
    )
    result = df.mean(numeric_only=True)

    result["Protocol Type"] = float(df["Protocol Type"].mode().iloc[0])

    for name in ("ack_count", "syn_count", "fin_count", "rst_count"):
        result[name] = float(df[name].sum())

    sizes = df["Tot size"]
    result["Tot sum"] = float(sizes.sum())
    result["Min"] = float(sizes.min())
    result["Max"] = float(sizes.max())
    result["AVG"] = float(sizes.mean())
    result["Std"] = float(sizes.std(ddof=1))
    result["Variance"] = float(sizes.var(ddof=1))
    result["Number"] = float(df["Number"].sum())

    duration = max(timestamps) - min(timestamps)
    result["Rate"] = float(len(df) / duration) if duration > 0 else 0.0

    result = result.reindex(FEATURES).replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0.0)

    return result


def capture_batches(iface="eth0", count=100, timeout=30):
    """Capture at most count packets and yield complete feature batches."""
    if not 1 <= count <= 1000:
        raise ValueError("count must be between 1 and 1000")

    captured_rows = []
    previous_timestamp = None
    batches = []
    seen = 0

    def handle(packet):
        nonlocal previous_timestamp, seen
        seen += 1

        row, ts = packet_features(packet, previous_timestamp)
        previous_timestamp = ts

        if row is not None:
            row["_timestamp"] = ts
            captured_rows.append(row)

        return seen >= count

    print(f"Capturing on {iface}: maximum {count} packets, timeout {timeout}s")
    sniff(iface=iface, prn=handle, store=False, count=count, timeout=timeout)

    for start in range(0, len(captured_rows) - BATCH_SIZE + 1, BATCH_SIZE):
        batch = captured_rows[start:start + BATCH_SIZE]
        batches.append(aggregate_batch(batch))

    output = pd.DataFrame(batches, columns=FEATURES)
    print(f"Packets captured: {seen}")
    print(f"IPv4/ARP-compatible packet rows: {len(captured_rows)}")
    print(f"Complete feature batches: {len(output)}")
    print(f"Discarded incomplete rows: {len(captured_rows) % BATCH_SIZE}")
    return output
