# PushT data tools

Downloads the research dataset `quentinll/lewm-pusht/pusht_expert_train.h5.zst`
to `/root/autodl-tmp/pusht`. The archive stays compressed. The remote filesystem's
reported free space, rather than the advertised host capacity, determines whether
it fits. Allow additional space before extracting or caching encoder features.

## Design and implementation plan

Use Hugging Face's resumable HTTP client, pin each download to a dataset commit,
then check file size and the published SHA-256. HTTP makes partial-file byte counts
observable; Xet is disabled for this workflow. Save provenance in a JSON manifest.
Keep credentials and data outside Git. Sync committed code using a Git bundle over
SSH, which does not require placing GitHub credentials on the GPU server.

1. Test acceptance of valid bytes and rejection of truncated/corrupt downloads.
2. Implement the downloader and AutoDL acceleration wrapper.
3. Commit only the data tools and ignore rules; transfer that commit to the server.
4. Run detached, monitor logs, and verify the archive before reporting success.

## Install and run on AutoDL

From the repository root, using your chosen Python environment:

```bash
python3 -m pip install -r scripts/data/requirements.txt
mkdir -p /root/autodl-tmp/pusht
nohup bash scripts/data/download_pusht.sh \
  --destination /root/autodl-tmp/pusht \
  > /root/autodl-tmp/pusht/download.log 2>&1 < /dev/null &
tail -f /root/autodl-tmp/pusht/download.log
```

The wrapper runs `source /etc/network_turbo` before network requests. Set
`DATA_PYTHON=/path/to/python` if Python is not on the SSH shell's PATH. Install
dependencies before enabling the proxy; AutoDL notes that it can slow pip mirrors.

Progress appears every 15 seconds with bytes, percentage, speed, and estimated time.
Rerun the same command after interruption to resume. A lock prevents concurrent
downloads to the same destination. Success requires a `VERIFIED` log line and
`download_manifest.json`, containing the source revision, code commit, size, and
SHA-256. A completed transfer alone is not verification. A corrupt existing archive
causes failure rather than being silently accepted or deleted.

## Verify locally

```bash
python3 -m unittest discover -s scripts/data/tests -v
bash -n scripts/data/download_pusht.sh
```

## Sync committed code

Create a bundle locally with `git bundle create /tmp/pusht-code.bundle HEAD`,
transfer it using `scp -P PORT`, then on the server use
`git clone /path/to/pusht-code.bundle /root/autodl-tmp/2883FinalProject`.
For subsequent bundles, fetch their `HEAD` and merge with `--ff-only`; never reset
remote work. Both checkouts should report the same `git rev-parse HEAD`.

The archive and the original research documents describe data that still needs
schema inspection before implementing the loader. Do not assume image dimensions,
episode counts, or simulator-state field order from estimates in the proposal.
