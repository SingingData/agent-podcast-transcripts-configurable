# corrections

This directory is the live runtime source for transcript correction dictionaries.

Layout:

- `global.txt`
  - corrections that apply across all shows
- `shows/<show-slug>.txt`
  - corrections that apply only to one show

Runtime behavior:

- global corrections load for every transcript and weekly summary
- per-show corrections load only for that show's episode
- per-show corrections override global corrections on conflicts

Format:

- `Canonical Phrase | misspelling 1, misspelling 2`
- `Phrase to delete |`

Guidelines:

- Prefer phrase-level corrections over broad single-word corrections
- Prefer exact branded strings for names, firms, URLs, and emails
- Keep delete rules rare and intentional
- If a correction is only relevant to one show, put it in that show's file
