import os
import re

source = "ssrn_working_paper.md"
dest_dir = "/Users/shreyasrkarwa/.gemini/antigravity/brain/e2bef2a9-329f-4004-9a7c-0c135667c713"

with open(source, "r") as f:
    content = f.read()

# Write SSRN version
ssrn_path = os.path.join(dest_dir, "ssrn_paper_final.md")
with open(ssrn_path, "w") as f:
    f.write(content)

# Strip author details for JRPM
# Lines 3 to 9 contain author info:
# **Shreyas Karwa**
# *Atlassian*
# **Date:** April 2026
# *Disclaimer...*
lines = content.split('\n')
jrpm_lines = [lines[0], lines[1]] + lines[10:]
jrpm_content = '\n'.join(jrpm_lines)

jrpm_path = os.path.join(dest_dir, "jrpm_blind_review.md")
with open(jrpm_path, "w") as f:
    f.write(jrpm_content)

print(f"Created {ssrn_path}")
print(f"Created {jrpm_path}")
