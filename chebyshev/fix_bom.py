import os

# Read current pyproject.toml as binary
with open("pyproject.toml", "rb") as f:
    content = f.read()

# Check for BOM
if content.startswith(b"\xef\xbb\xbf"):
    print("Found UTF-8 BOM - removing it...")
    content = content[3:]
else:
    print("No BOM found")

# Write back without BOM
with open("pyproject.toml", "wb") as f:
    f.write(content)

print(f"Fixed pyproject.toml ({len(content)} bytes)")

# Verify first few bytes
with open("pyproject.toml", "rb") as f:
    header = f.read(20)
    print(f"First 20 bytes: {header}")
