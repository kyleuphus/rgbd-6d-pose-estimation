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
# The snapshot contains older source code and sample inputs. The run cell pulls your latest committed and pushed source by default; it does not download additional dataset frames. Set **UPDATE_SOURCE = False** only when the restored source already accepts the current command-line arguments.
#
# Each case uses the existing single-frame script in the cached Python/CUDA environment. The backend is initialized separately for each case. Failed cases are recorded and the remaining cases continue. The final cell derives the download contents from this run's successful cases.


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
# 4. Run all selected cases in the saved Python/CUDA environment.
import json
import shlex
import subprocess
from datetime import datetime, timezone

if UPDATE_SOURCE:
    subprocess.run(
        ['git', '-C', str(PROJECT_ROOT), 'pull', '--ff-only'],
        check=True,
    )

# Keep inference in the restored environment, not the notebook's Python kernel.
SETUP = """set -eo pipefail
source /content/foundationpose-setup.sh > /dev/null
export PYTHONPATH='/content/uwrt/FoundationPose/mycpp/build:/content/uwrt/FoundationPose'
cd /content/uwrt/FoundationPose
"""
gpu_check = "import torch; assert torch.cuda.is_available(), 'Select a T4 GPU runtime'"
subprocess.run(
    ['bash', '-c', SETUP + shlex.join(['python', '-c', gpu_check])],
    check=True,
)

run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
BATCH_DIR = PROJECT_ROOT / 'outputs' / 'batches' / run_id
BATCH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = BATCH_DIR / 'results.json'
RESULTS = []

for index, case in enumerate(CASES, start=1):
    case_name = (
        f"{case['folder_id']}_{case['image_id']}_obj_{case['object_id']:06d}"
    )
    pose_path = PROJECT_ROOT / 'outputs' / case_name / f'pose_{case_name}.npy'
    command = [
        'python', str(PROJECT_ROOT / 'src/run_frame.py'),
        '--folder-id', case['folder_id'],
        '--image-id', case['image_id'],
        '--object-id', str(case['object_id']),
    ]
    print(f'\n[{index}/{len(CASES)}] {case_name}', flush=True)
    completed = subprocess.run(['bash', '-c', SETUP + shlex.join(command)])

    success = completed.returncode == 0 and pose_path.is_file()
    result = {
        **case,
        'status': 'ok' if success else 'failed',
        'returncode': completed.returncode,
        'pose_path': str(pose_path) if success else None,
    }
    if not success:
        result['error'] = (
            f'Inference exited with code {completed.returncode}.'
            if completed.returncode != 0
            else f'Expected prediction was not created: {pose_path}'
        )
    RESULTS.append(result)
    # Save progress after each case, including failures.
    RESULTS_PATH.write_text(json.dumps(RESULTS, indent=2) + '\n', encoding='utf-8')
    print(f"  {result['status']}", flush=True)

successful = sum(result['status'] == 'ok' for result in RESULTS)
print(f'\nFinished: {successful}/{len(RESULTS)} succeeded.')
print(f'Run summary: {RESULTS_PATH}')


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
    for result in successful_results:
        pose_path = Path(result['pose_path'])
        archive.write(
            pose_path,
            arcname=str(pose_path.relative_to(PROJECT_ROOT / 'outputs')),
        )

print(f'Downloading {len(successful_results)} prediction(s) and the run summary.')
files.download(str(archive_path))


