import re

# Input/output paths
input_file = "prepare/buggy_program/bug_list.txt"
output_file = "bugs.tsv"

with open(input_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

with open(output_file, "w") as out:
    out.write("project\tid\tkind\n")
    for line in lines:
        # Example: "Chart-11" or "Time-25"
        match = re.match(r"([A-Za-z]+)-(\d+)", line)
        if match:
            project, bug_id = match.groups()
            # Write two lines: one buggy, one fixed
            out.write(f"{project}\t{bug_id}\tb\n")
            out.write(f"{project}\t{bug_id}\tf\n")
        else:
            print(f"⚠️ Skipped invalid line: {line}")

print(f"✅ Converted {len(lines)} entries from {input_file} → {output_file}")
