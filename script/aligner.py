import re
import sys
import argparse
from datetime import datetime, timedelta

def parse_conpot_time(ts_str):
    """Parses 'YYYY-MM-DD HH:MM:SS,mmm' into a datetime object."""
    # Split "2025-12-13 11:58:47,355" into main time and millis
    main_part, millis = ts_str.split(',')
    dt = datetime.strptime(main_part, "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(microsecond=int(millis) * 1000)
    return dt

def format_conpot_time(dt):
    """Formats datetime object back to 'YYYY-MM-DD HH:MM:SS,mmm'."""
    main_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    millis = int(dt.microsecond / 1000)
    return f"{main_str},{millis:03d}"

def align_logs(input_file, output_file):
    print(f"[-] Reading mixed logs from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    if not lines:
        print("Error: Empty file.")
        return

    aligned_lines = []
    
    # 1. Find the first valid timestamp to start the clock
    last_valid_dt = None
    
    # Regex to find timestamp at start of line
    ts_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})')

    corrections = 0
    
    for line in lines:
        line = line.strip()
        match = ts_pattern.match(line)
        
        if match:
            current_ts_str = match.group(1)
            current_dt = parse_conpot_time(current_ts_str)
            
            if last_valid_dt is None:
                # First line sets the baseline
                last_valid_dt = current_dt
                new_line = line
            else:
                # Calculate time difference
                diff = current_dt - last_valid_dt
                
                # LOGIC: 
                # If the time jumps forward more than 1 hour (likely an injection from another day)
                # OR if the time jumps BACKWARDS (injection from the past)
                # THEN: We overwrite the timestamp to be (Previous + 10ms)
                
                is_gap = diff > timedelta(hours=1)
                is_backward = diff.total_seconds() < 0
                
                if is_gap or is_backward:
                    # Stitch it! 
                    # Add 10 milliseconds to the last valid time
                    stitched_dt = last_valid_dt + timedelta(milliseconds=10)
                    
                    # Create new timestamp string
                    new_ts_str = format_conpot_time(stitched_dt)
                    
                    # Replace in the line string
                    new_line = line.replace(current_ts_str, new_ts_str, 1)
                    
                    # Update our tracking variable
                    last_valid_dt = stitched_dt
                    corrections += 1
                else:
                    # Normal flow (e.g. 2 seconds polling), keep it.
                    last_valid_dt = current_dt
                    new_line = line
            
            aligned_lines.append(new_line)
        else:
            # Lines without timestamps (e.g. traces) just get appended
            aligned_lines.append(line)

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(aligned_lines))
        f.write('\n') # Ensure EOF newline

    print(f"[+] Alignment Complete.")
    print(f"    - Output saved to: {output_file}")
    print(f"    - Total Lines: {len(lines)}")
    print(f"    - Timestamps Corrected (Stitched): {corrections}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_log', help="The messy mixed log file")
    parser.add_argument('output_log', help="The clean continuous log file")
    args = parser.parse_args()
    
    align_logs(args.input_log, args.output_log)