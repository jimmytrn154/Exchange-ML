# Segmentation backend

The current experiments reuse PENGWIN ABBC/STU-Net. The checkout path is
`baselines/pengwin2026-task1-abbc/`, pinned to upstream commit
`ccc625fd000ec2dfce622b4621bf2d0b4ee12796`, with the project patch at
[`../patches/pengwin2026-task1-abbc_on_ccc625f.patch`](../patches/pengwin2026-task1-abbc_on_ccc625f.patch).

The local backend was restored at this commit and patched for E3. The E3
preflight verifies the source bytes before inference. The nnU-Net Python
package version is pinned in the environment freeze. See the [E3 protocol](../__docs__/exps.md#e3-held-out-benchmark--thực-nghiệm-chính).

Unused active-learning reference placeholders were removed during cleanup.
Their Git commit references are retained in the cleanup backup outside this repo.
