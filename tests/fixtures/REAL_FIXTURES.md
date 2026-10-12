# REAL mainnet fixtures (unmodified RPC responses)

Each file is the byte-for-byte JSON-RPC response body (`{"jsonrpc","result","id"}`)
returned by a public read-only call. Nothing was edited, relabelled or re-serialised.
Only landed, finalized public transactions; no keys, nothing signed or sent.

Fetch: `POST https://api.mainnet-beta.solana.com`
`getTransaction [<signature>, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0, "commitment": "finalized"}]`
Candidates were found with `getSignaturesForAddress` on Jupiter v6
`JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4` (finalized), fetched 2026-10-12.

| file | kind | signature | slot | blockTime (UTC) | sha256 |
|---|---|---|---|---|---|
| `real_token2022_buy_41WvNWct.json` | REAL | `41WvNWctkG8GQQ9w3tDMG1uQA1upTGtMqYx5Eu5dEKenQQkthdH2RwLX2QPmRFMg8jQbh3CZjmbzqhinVGaNuzmy` | 456150440 | 1791830829 (2026-10-12T18:47:09Z) | `5c4a501481daa433409f3506e16b7b9a9bac532d3c24bcd7a97a6ccad7b40eae` |
| `real_spl_buy_4GVYwAeQ.json` | REAL | `4GVYwAeQuRwzWirFyrgXneJro7SHcYsvF3cVWa6YVUBHFHedzAMKsqiBDQghH4EqDK3hefrsHu7UWV7dGA8dN4gV` | 456122647 | 1791824744 (2026-10-12T17:05:44Z) | `ce5fb194b2f49a40ecaa57a590cd4478ef27368d05ddc9d051fabb8be06e8b7b` |

Explorer: `https://solscan.io/tx/<signature>`.

* `41WvNWct`: native SOL -> Token-2022 token `J6Y3vyed8iStNBwcHyoc3ct63kd9SsxsTainjgNYzQJ4`
  (6 decimals, no transfer-fee movement) by `FcQQtjEGTNRXDJQ6e6mtAHPe6syW2wdF9ENN8T6T9fZQ`.
  Note: jsonParsed labels its Token-2022 instructions `"program": "spl-token"`
  with `programId` `TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb`.
* `4GVYwAeQ`: native SOL -> SPL Token `85VBFQZC9TZkfaptBWjvUw7YbZjy52A6mjtPGjstQAmQ`
  (6 decimals) by `7hARSicdYvWR9saun1F3JDQjnnHY3k6wvNH2XXp2EA6A`.

Expected amounts are derived independently in
`tests/test_p1_token2022_jsonparsed_label.py` (see its docstrings).
