# Jean-Zay setup

Setting up the environment and downloading the model on Jean-Zay.

`main.sh` launches the script on the A100 partition.
`volta.sh` launches the script on the V100 partition.

To chain jobs, use the `--dependency` option of `sbatch`. For example, to launch `volta.sh` after `main.sh` completes successfully:

```bash
jobid=$(sbatch --parsable main.sh)
sbatch --dependency=afterok:$jobid volta.sh
```