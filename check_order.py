with open('app.py', encoding='utf-8') as f:
    lines = f.readlines()

print('Total lines:', len(lines))
keywords = ['__main__', 'app.run(', 'def select_role', 'def interview_page',
            'def recruiter_dashboard_page', 'def admin_dashboard_page', 'if False:']
for i, line in enumerate(lines, 1):
    s = line.strip()
    if any(k in s for k in keywords):
        print(i, repr(s[:90]))
