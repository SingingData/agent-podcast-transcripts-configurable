# correction-requests

This folder stores local correction submissions captured from reply emails sent in response to transcript emails.

## File layout

One file is maintained per show:

- `corrections-submission-animal-spirits.txt`
- `corrections-submission-the-compound-and-friends.txt`
- `corrections-submission-ask-the-compound.txt`
- `corrections-submission-masters-in-business.txt`
- `corrections-submission-at-the-money.txt`

## Submission format

Each accepted correction is appended as one `|`-separated line:

```txt
Episode Title|Sender Name|correction: wrong phrase \ correct phrase
```

Newlines and literal pipe characters in stored fields are escaped before writing.

## Git behavior

The per-show submission `.txt` files are ignored by git.
This `README.md` file is not ignored and may be committed.
