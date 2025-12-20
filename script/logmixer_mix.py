import re
import random
import itertools
import os

def parse_blocks_by_second(filepath):
    """
    Parses a log file into blocks of lines grouped by the second of the timestamp.
    This ensures multi-line events (like a connection + request + response) stay together.
    """
    blocks = []
    current_block = []
    current_timestamp = None
    # Regex looks for "YYYY-MM-DD HH:MM:SS" at the start of the line
    timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return []

    print(f"Parsing blocks from {filepath}...")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = timestamp_pattern.match(line)
            if match:
                timestamp = match.group(1)
                # If timestamp changes (new second), close current block and start new one
                if timestamp != current_timestamp:
                    if current_block:
                        blocks.append(current_block)
                    current_block = [line]
                    current_timestamp = timestamp
                else:
                    # Same timestamp, append to current block
                    current_block.append(line)
            else:
                # No timestamp (e.g., error trace), append to current block
                if current_block:
                    current_block.append(line)
                else:
                    # Orphan line at start
                    blocks.append([line])
    
    # Append the last block
    if current_block:
        blocks.append(current_block)
        
    return blocks

def filter_blocks(blocks, keyword):
    """
    Keeps only blocks where at least one line contains the keyword.
    """
    filtered = []
    for block in blocks:
        if any(keyword in line for line in block):
            filtered.append(block)
    return filtered

def mix_three_logs(normal_file, attack_file, noise_file, output_file):
    # --- Configuration Ratios ---
    NORMAL_RATIO = 0.90       # 90% of total lines are Normal
    
    # Of the remaining 10% (Abnormal), how do we split them?
    ABNORMAL_ATTACK_RATIO = 0.70  # 70% Attack
    ABNORMAL_NOISE_RATIO = 0.30   # 30% Noise

    # 1. Parse all logs using timestamp blocking
    norm_blocks = parse_blocks_by_second(normal_file)
    att_blocks = parse_blocks_by_second(attack_file)
    noise_blocks_raw = parse_blocks_by_second(noise_file)
    
    # 2. Filter Noise Log (Only keep blocks with 172.19.0.1)
    noise_blocks = filter_blocks(noise_blocks_raw, "172.19.0.1")
    
    if not norm_blocks:
        print("Error: Normal log is empty.")
        return

    # 3. Calculate Targets based on Normal Log length
    n_count = len(norm_blocks)
    
    # Formula: N / (N + Abnormal) = 0.90  =>  Abnormal = N * (1 - 0.9) / 0.9
    if NORMAL_RATIO >= 1.0:
        total_abnormal = 0
    else:
        total_abnormal = int(n_count * (1 - NORMAL_RATIO) / NORMAL_RATIO)
        
    # Split the total abnormal count into Attack vs Noise
    target_attack_count = int(total_abnormal * ABNORMAL_ATTACK_RATIO)
    target_noise_count = int(total_abnormal * ABNORMAL_NOISE_RATIO)
    
    # Adjust for rounding errors if necessary (ensure sum equals total_abnormal)
    # optional: target_noise_count = total_abnormal - target_attack_count 

    print(f"\n--- Statistics ---")
    print(f"Normal Blocks Found: {n_count}")
    print(f"Target Total Abnormal Blocks: {total_abnormal} (to reach ~10% of total)")
    print(f"  -> Target Attack Blocks: {target_attack_count} (70% of abnormal)")
    print(f"  -> Target Noise Blocks:  {target_noise_count} (30% of abnormal)")

    # 4. Prepare Pools (Cycle/Repeat if source is too short)
    final_attack_sequence = []
    if att_blocks:
        att_cycle = itertools.cycle(att_blocks)
        final_attack_sequence = [next(att_cycle) for _ in range(target_attack_count)]
    else:
        print("Warning: Attack log is empty.")

    final_noise_sequence = []
    if noise_blocks:
        noise_cycle = itertools.cycle(noise_blocks)
        final_noise_sequence = [next(noise_cycle) for _ in range(target_noise_count)]
    else:
        print("Warning: No noise blocks matched IP 172.19.0.1. Noise portion will be 0.")

    # 5. Create Mask and Shuffle
    # 'N' = Normal
    # 'A' = Attack
    # 'O' = Noise (Other)
    mask = ['N'] * n_count + ['A'] * len(final_attack_sequence) + ['O'] * len(final_noise_sequence)
    random.shuffle(mask)
    
    # 6. Interleave based on the shuffled mask
    mixed_lines = []
    iter_norm = iter(norm_blocks)
    iter_att = iter(final_attack_sequence)
    iter_noise = iter(final_noise_sequence)
    
    for kind in mask:
        if kind == 'N':
            mixed_lines.extend(next(iter_norm))
        elif kind == 'A':
            mixed_lines.extend(next(iter_att))
        elif kind == 'O':
            mixed_lines.extend(next(iter_noise))
            
    # 7. Write Result
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(mixed_lines)
    
    print(f"\nSuccess! Generated '{output_file}' with {len(mixed_lines)} lines.")


# --- Configuration ---
normal_log_path = 'normal.log'
attack_log_path = 'pure_attack.log'
noise_log_path = 'pure_noise.log'
output_log_path = 'attack.log'

# --- Run Configuration ---
if __name__ == "__main__":
    # Ensure these filenames match your actual files
    mix_three_logs(normal_log_path, attack_log_path, noise_log_path, output_log_path)