"""
fix_routes.py
Fixes the root cause of /select-role 404:
  - The multi-domain routes were defined AFTER app.run() inside __main__
  - This script disables the premature app.run() and adds it at the very end
"""
import os

path = os.path.join(os.path.dirname(__file__), 'app.py')

# Detect encoding
with open(path, 'rb') as f:
    raw = f.read(3)

if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
    enc = 'utf-16'
elif raw[:3] == b'\xef\xbb\xbf':
    enc = 'utf-8-sig'
else:
    enc = 'utf-8'

print(f'Detected encoding: {enc}')

with open(path, 'r', encoding=enc) as f:
    content = f.read()

TARGET = "if __name__ == '__main__':"
REPLACEMENT = "if False:  # app.run moved to end of file so all routes register first"

if TARGET not in content:
    print('ERROR: Could not find __main__ block. Already fixed or different format.')
else:
    # Replace only the FIRST occurrence (the premature one before multi-domain routes)
    content = content.replace(TARGET, REPLACEMENT, 1)
    print('Replaced premature __main__ block.')

# Remove any existing "# FIXED MAIN" block we may have added before
FIXED_MARKER = '\n# ===== FIXED MAIN ENTRY POINT ====='
if FIXED_MARKER in content:
    idx = content.index(FIXED_MARKER)
    content = content[:idx]
    print('Removed previous fixed main block.')

# Append the real __main__ at the very end
content += """

# ===== FIXED MAIN ENTRY POINT =====
# app.run() is here so ALL routes (including multi-domain) are registered first
if __name__ == '__main__':
    print("\\n" + "="*55)
    print("  EmotionLearnAI - Multi-Domain Platform")
    print("="*55)
    import os as _os
    _gemini = _os.getenv('GEMINI_API_KEY', '')
    print(f"  Gemini AI   : {'[ON]  Enabled' if _gemini else '[OFF] Set GEMINI_API_KEY in .env'}")
    print("  URLs        : http://localhost:5000")
    print("              : http://localhost:5000/select-role")
    print("              : http://localhost:5000/interview")
    print("              : http://localhost:5000/skill-assessment")
    print("              : http://localhost:5000/employee-training")
    print("              : http://localhost:5000/recruiter-dashboard")
    print("              : http://localhost:5000/admin-dashboard")
    print("="*55 + "\\n")
    app.run(debug=True, port=5000)
"""

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done. app.py fixed. Now run: python app.py')
