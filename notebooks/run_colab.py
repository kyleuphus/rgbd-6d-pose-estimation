# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FoundationPose: daily evaluation runs
# Select **Runtime → Change runtime type → T4 GPU**. Mount Drive, restore the saved setup, select cases, run inference, then download the results.
#
# Edit only **CASE_GROUPS** to choose your folders, images, and objects. Each group runs every selected image/object combination within that folder. Use separate groups when different images contain different objects. IDs can be integers or strings, with or without leading zeros. Cases listed more than once run only once.
#
# The snapshot contains older source code and sample inputs. The run cell pulls your latest committed and pushed source by default; it does not download additional dataset frames. Commit and push the updated batch runner, backend, and data loaders before running. Set **UPDATE_SOURCE = False** only when those changes are already present in the runtime.
#
# The batch runs in one Python process in the cached Python/CUDA environment. Model weights and the renderer load once; each object's mesh and backend are prepared once and reused across its frames. Cases run grouped by object, with independent registration on every frame (no tracking). GPU-synchronized timings separate imports, model/object setup, data loading, registration, and saving. The first registration for each object is flagged because it may include warm-up. Failed cases are recorded and the remaining cases continue. The final cell downloads successful predictions, timings, selected cases, and the console log.


# %%
# 1. Mount Drive.
from google.colab import drive
drive.mount('/content/drive')

# %%
%%bash
# 2. Restore the working setup on a fresh runtime.
set -eo pipefail

if [ -x /content/foundationpose-env/bin/python ] && \
   [ -f /content/foundationpose-setup.sh ] && \
   [ -f /content/uwrt/pose-estimation/src/run_frame.py ]; then
    echo 'Using the existing setup.'
else
    cp /content/drive/MyDrive/uwrt-pose-cache/foundationpose-working.tar.gz \
        /content/foundationpose-working.tar.gz
    tar -xzf /content/foundationpose-working.tar.gz -C /
    echo 'Saved setup restored.'
fi
mkdir -p /content/mamba

# %%
# 3. Choose cases here. This selection also controls the download.
from pathlib import Path

PROJECT_ROOT = Path('/content/uwrt/pose-estimation')
UPDATE_SOURCE = True

CASE_GROUPS = [
    {
        'folder_id': '000048',
        'image_ids': ['000001', '001074'],
        'object_ids': [1],
    },
    # Add another dictionary for another folder or a different set of objects.
]

CASES = []
for group in CASE_GROUPS:
    if not group['image_ids'] or not group['object_ids']:
        raise ValueError('Each group needs at least one image and one object.')
    folder_number = int(group['folder_id'])
    for image_id in group['image_ids']:
        for object_id in group['object_ids']:
            image_number = int(image_id)
            object_number = int(object_id)
            if folder_number < 0 or image_number < 0 or object_number <= 0:
                raise ValueError('Folder/image IDs must be nonnegative; object IDs must be positive.')
            case = {
                'folder_id': f'{folder_number:06d}',
                'image_id': f'{image_number:06d}',
                'object_id': object_number,
            }
            if case not in CASES:
                CASES.append(case)

if not CASES:
    raise ValueError('Select at least one case in CASE_GROUPS.')

print(f'Selected {len(CASES)} case(s):')
for case in CASES:
    print(f"  folder {case['folder_id']}, image {case['image_id']}, object {case['object_id']}")


# %%
# 4. Run the batch once, sharing model weights and each object's backend.
import json
import shlex
import subprocess
from datetime import datetime, timezone

if UPDATE_SOURCE:
    subprocess.run(
        ['git', '-C', str(PROJECT_ROOT), 'pull', '--ff-only'],
        check=True,
    )

runner = PROJECT_ROOT / 'src/run_batch.py'
if not runner.is_file():
    raise FileNotFoundError(
        'run_batch.py is missing. Commit and push the updated source, then rerun this cell.'
    )

run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
BATCH_DIR = PROJECT_ROOT / 'outputs' / 'batches' / run_id
BATCH_DIR.mkdir(parents=True, exist_ok=True)
CASES_PATH = BATCH_DIR / 'cases.json'
RESULTS_PATH = BATCH_DIR / 'results.json'
TIMINGS_PATH = BATCH_DIR / 'timings.json'
LOG_PATH = BATCH_DIR / 'batch.log'
CASES_PATH.write_text(json.dumps(CASES, indent=2) + '\n', encoding='utf-8')
RESULTS = []

# Run in the restored environment, with unbuffered live progress.
setup = """set -eo pipefail
source /content/foundationpose-setup.sh > /dev/null
export PYTHONPATH='/content/uwrt/FoundationPose/mycpp/build:/content/uwrt/FoundationPose'
cd /content/uwrt/FoundationPose
"""
command = [
    'python', '-u', str(runner),
    '--project-root', str(PROJECT_ROOT),
    '--cases-file', str(CASES_PATH),
    '--results-file', str(RESULTS_PATH),
]
print(f'Running {len(CASES)} case(s) in one Python process.', flush=True)
with LOG_PATH.open('w', encoding='utf-8') as log:
    process = subprocess.Popen(
        ['bash', '-c', setup + 'exec ' + shlex.join(command)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    except KeyboardInterrupt:
        # The shell uses exec, so this terminates the actual inference process.
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print('Stopped. Completed cases remain available in the download cell.')
        raise
    finally:
        process.stdout.close()
        if RESULTS_PATH.is_file():
            RESULTS = json.loads(RESULTS_PATH.read_text(encoding='utf-8'))

if returncode != 0:
    raise RuntimeError(f'Batch exited with code {returncode}; see the output above or {LOG_PATH}.')
print(f'Batch files: {BATCH_DIR}')


# %%
# 5. Download this run's predictions and summary as one ZIP (optional).
from zipfile import ZipFile, ZIP_DEFLATED
from google.colab import files

if 'RESULTS' not in globals() or not RESULTS:
    raise RuntimeError('Run the inference cell before downloading results.')

successful_results = [result for result in RESULTS if result['status'] == 'ok']
archive_path = BATCH_DIR / 'predictions.zip'
with ZipFile(archive_path, 'w', compression=ZIP_DEFLATED) as archive:
    archive.write(RESULTS_PATH, arcname='results.json')
    for extra_path in (TIMINGS_PATH, CASES_PATH, LOG_PATH):
        if extra_path.is_file():
            archive.write(extra_path, arcname=extra_path.name)
    for result in successful_results:
        pose_path = Path(result['pose_path'])
        archive.write(
            pose_path,
            arcname=str(pose_path.relative_to(PROJECT_ROOT / 'outputs')),
        )

print(f'Downloading {len(successful_results)} prediction(s) and the run summary.')
files.download(str(archive_path))


