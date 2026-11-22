from pathlib import Path

p = Path('README.md')
s = p.read_text(encoding='utf-8')
s = s.replace('## Submission release v16 (2026-11-21)', '## Submission release v19 (2026-11-22)')
s = s.replace('## Submission release v17 (2026-11-21)', '## Submission release v19 (2026-11-22)')
s = s.replace('## Submission release v18 (2026-11-21)', '## Submission release v19 (2026-11-22)')
s = s.replace('fmg-submission-release-v16-2026-11-21', 'fmg-submission-release-v19-2026-11-22')
s = s.replace('fmg-submission-release-v17-2026-11-21', 'fmg-submission-release-v19-2026-11-22')
s = s.replace('fmg-submission-release-v18-2026-11-21', 'fmg-submission-release-v19-2026-11-22')
s = s.replace('## Submission release v19 (2026-11-22)', '## Submission release v20 (2026-11-22)')
s = s.replace('fmg-submission-release-v19-2026-11-22', 'fmg-submission-release-v20-2026-11-22')
p.write_text(s, encoding='utf-8')
print('README updated to v20')
