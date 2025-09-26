import re

# Path to your log file
log_file = "raw.txt"

# Threshold for filtering
threshold = 0.9655

with open(log_file, "r", encoding="utf-8") as f:
    log_text = f.read()

pattern = re.compile(r"(di-[A-Za-z0-9-]+)\s*\n\s*([0-9]*\.[0-9]+)")
matches = pattern.findall(log_text)

filtered_ids = [di_id for di_id, val in matches if float(val) <= threshold]

dicom_list = [
    "di-0CA9-8FC9-25BB", "di-1F8E-37E5-57ED", "di-1FD4-CB18-6EFC", "di-5A75-9C41-46D7", "di-5BB1-44DE-60BA",
    "di-62CF-F093-A156", "di-63D5-7095-7FB9", "di-A540-DBDD-7C1F", "di-AE68-A41B-5185", "di-C910-B188-A16F",
    "di-EB49-9AE0-F0D0", "di-6CF6-853C-CB97", "di-16B9-402D-037F", "di-46D0-7328-762F", "di-47EB-1516-2456",
    "di-7041-D238-665F", "di-8254-6AD3-C7FC", "di-A984-8D28-57F4", "di-B55F-84D9-6833", "di-C9E0-1668-D365",
    "di-E42E-19EA-16E7"
]

# Compute overlap
overlap = sorted(set(filtered_ids) & set(dicom_list))

print(f"Found {len(filtered_ids)} IDs above threshold {threshold}")
print("filtered_ids =", filtered_ids)

print(f"\nOverlap with dicom_list ({len(overlap)} IDs):")
print("overlap =", overlap)
